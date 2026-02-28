import express from 'express';
import fs from 'fs';
import path from 'path';
import { execFileSync, execSync } from 'child_process';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = process.env.PORT || 3472;

function loadConfig() {
  const p = path.join(__dirname, 'config', 'run.json');
  return JSON.parse(fs.readFileSync(p, 'utf8'));
}

function runLocal(cmd) {
  return execSync(cmd, { encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'] });
}

function runRemote(host, cmd) {
  return execFileSync('ssh', [host, cmd], { encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'] });
}

function safeExec(cfg, cmd) {
  try {
    if (cfg.host === 'local') return runLocal(cmd);
    return runRemote(cfg.host, cmd);
  } catch (e) {
    return '';
  }
}

function parseLog(logText) {
  const steps = [];
  const checkpoints = [];
  let completed = false;

  for (const line of logText.split('\n')) {
    const m = line.match(/step=(\d+)\s+loss=([0-9.]+)/);
    if (m) {
      steps.push({ step: Number(m[1]), loss: Number(m[2]) });
      continue;
    }
    const c = line.match(/checkpoint_saved=(.+head_step_(\d+)\.pt)/);
    if (c) {
      checkpoints.push({ path: c[1], step: Number(c[2]) });
      continue;
    }
    if (line.includes('SMOKE_OK')) completed = true;
  }

  return { steps, checkpoints, completed };
}

function rollingAvg(points, window = 10) {
  const out = [];
  let sum = 0;
  for (let i = 0; i < points.length; i++) {
    sum += points[i].loss;
    if (i >= window) sum -= points[i - window].loss;
    const denom = Math.min(i + 1, window);
    out.push({ step: points[i].step, loss: Number((sum / denom).toFixed(6)) });
  }
  return out;
}

function stats(points) {
  if (!points.length) {
    return {
      latestStep: 0,
      latestLoss: null,
      minLoss: null,
      maxLoss: null,
      p95Loss: null,
      spikeCount: 0,
      lowLossStreak: 0,
    };
  }

  const losses = points.map(p => p.loss).slice().sort((a, b) => a - b);
  const latest = points[points.length - 1];
  const min = losses[0];
  const max = losses[losses.length - 1];
  const p95 = losses[Math.floor(0.95 * (losses.length - 1))];
  const spikeCount = points.filter(p => p.loss > 0.1).length;

  let lowLossStreak = 0;
  for (let i = points.length - 1; i >= 0; i--) {
    if (points[i].loss <= 0.01) lowLossStreak += 1;
    else break;
  }

  return {
    latestStep: latest.step,
    latestLoss: latest.loss,
    minLoss: min,
    maxLoss: max,
    p95Loss: p95,
    spikeCount,
    lowLossStreak,
  };
}

function parseGpu(text) {
  return text
    .split('\n')
    .map(l => l.trim())
    .filter(Boolean)
    .map(line => {
      const [index, util, memUsed, memTotal, temp, power] = line.split(',').map(x => x.trim());
      return {
        index: Number(index),
        util: Number(util),
        memUsed: Number(memUsed),
        memTotal: Number(memTotal),
        temp: Number(temp),
        power: Number(power),
      };
    })
    .filter(x => Number.isFinite(x.index));
}

app.get('/api/metrics', (_req, res) => {
  const cfg = loadConfig();

  const grepCmd = `grep -E 'step=[0-9]+ loss=|checkpoint_saved=|SMOKE_OK' ${cfg.logPath} 2>/dev/null || true`;
  const procCmd = `pgrep -af '${cfg.processPattern}' || true`;
  const gpuCmd = "nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits";
  const mtimeCmd = `stat -c '%y' ${cfg.logPath} 2>/dev/null || true`;

  const logText = safeExec(cfg, grepCmd);
  const procText = safeExec(cfg, procCmd);
  const gpuText = safeExec(cfg, gpuCmd);
  const mtime = safeExec(cfg, mtimeCmd).trim();

  const parsed = parseLog(logText);
  const roll10 = rollingAvg(parsed.steps, 10);
  const roll50 = rollingAvg(parsed.steps, 50);
  const s = stats(parsed.steps);

  const checkpoints = parsed.checkpoints.length
    ? parsed.checkpoints
    : (safeExec(cfg, `ls -1 ${cfg.checkpointDir}/head_step_*.pt 2>/dev/null | sed 's#.*/head_step_##' | sed 's/.pt$//'`)
        .split('\n')
        .map(x => x.trim())
        .filter(Boolean)
        .map(x => ({ step: Number(x), path: `${cfg.checkpointDir}/head_step_${x}.pt` })));

  const active = procText.trim().length > 0;
  const progressPct = cfg.maxSteps ? Number(((s.latestStep / cfg.maxSteps) * 100).toFixed(2)) : 0;

  res.json({
    config: cfg,
    status: active ? 'active' : (parsed.completed ? 'completed' : 'idle'),
    logMtime: mtime,
    processLines: procText.split('\n').filter(Boolean),
    summary: {
      latestStep: s.latestStep,
      latestLoss: s.latestLoss,
      progressPct,
      minLoss: s.minLoss,
      maxLoss: s.maxLoss,
      p95Loss: s.p95Loss,
      spikeCount: s.spikeCount,
      lowLossStreak: s.lowLossStreak,
      checkpointCount: checkpoints.length,
      latestCheckpoint: checkpoints.length ? checkpoints[checkpoints.length - 1] : null,
    },
    series: {
      loss: parsed.steps,
      rolling10: roll10,
      rolling50: roll50,
      checkpoints,
    },
    gpus: parseGpu(gpuText).filter(g => cfg.gpus.includes(g.index)),
  });
});

app.use('/', express.static(path.join(__dirname, 'web')));

app.listen(PORT, () => {
  console.log(`[GemmPali Dashboard] listening on http://0.0.0.0:${PORT}`);
});
