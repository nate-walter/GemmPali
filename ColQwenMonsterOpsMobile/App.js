import React, { useCallback, useEffect, useMemo, useState } from 'react';
import {
  SafeAreaView,
  ScrollView,
  StyleSheet,
  Text,
  View,
  RefreshControl,
  StatusBar,
} from 'react-native';
import Svg, { Polyline, Line } from 'react-native-svg';

const API_URL = 'http://10.46.150.108:3473/api/metrics';
const REFRESH_MS = 5000;

function normalizeSeries(points = []) {
  const clean = (points || [])
    .map((p) => ({ step: Number(p?.step ?? 0), loss: Number(p?.loss ?? 0) }))
    .filter((p) => Number.isFinite(p.step) && Number.isFinite(p.loss))
    .sort((a, b) => a.step - b.step);

  if (!clean.length) return [];
  if (clean[0].step > 0) {
    return [{ step: 0, loss: clean[0].loss }, ...clean];
  }
  return clean;
}

function sampleByStep(points = [], maxPoints = 70) {
  if (!points?.length || points.length <= maxPoints) return points;
  const stride = Math.ceil(points.length / maxPoints);
  const sampled = points.filter((_, i) => i % stride === 0);
  const last = points[points.length - 1];
  if (!sampled.length || sampled[sampled.length - 1].step !== last.step) sampled.push(last);
  return sampled;
}

function smooth(values = [], alpha = 0.24) {
  if (!values.length) return values;
  const out = [];
  let prev = Number(values[0] || 0);
  for (const v of values) {
    const cur = Number(v ?? prev);
    prev = alpha * cur + (1 - alpha) * prev;
    out.push(prev);
  }
  return out;
}

function toPolylinePoints(stepLoss = [], width = 320, height = 110, pad = 6) {
  if (!stepLoss.length) return '';
  const losses = stepLoss.map((p) => Number(p.loss ?? 0));
  const min = Math.min(...losses);
  const max = Math.max(...losses);
  const span = Math.max(max - min, 1e-6);
  const maxStep = Math.max(stepLoss[stepLoss.length - 1].step, 1);

  return stepLoss
    .map((p) => {
      const x = pad + (Math.max(0, p.step) / maxStep) * (width - pad * 2);
      const y = height - pad - ((p.loss - min) / span) * (height - pad * 2);
      return `${x},${y}`;
    })
    .join(' ');
}

function KPI({ label, value, sub, accent = '#ff8f3f' }) {
  return (
    <View style={styles.kpiCard}>
      <Text style={styles.kpiLabel}>{label}</Text>
      <Text style={[styles.kpiValue, { color: accent }]} numberOfLines={1}>
        {value}
      </Text>
      <Text style={styles.kpiSub} numberOfLines={2}>{sub}</Text>
    </View>
  );
}

function SparkCard({ title, hint, series, color = '#ff8f3f' }) {
  const normalized = useMemo(() => normalizeSeries(series || []), [series]);
  const sampled = useMemo(() => sampleByStep(normalized, 72), [normalized]);
  const sampledVals = useMemo(() => sampled.map((p) => Number(p.loss ?? 0)), [sampled]);
  const smoothedVals = useMemo(() => smooth(sampledVals), [sampledVals]);
  const smoothedSeries = useMemo(
    () => sampled.map((p, i) => ({ step: p.step, loss: Number(smoothedVals[i] ?? p.loss ?? 0) })),
    [sampled, smoothedVals]
  );
  const points = useMemo(() => toPolylinePoints(smoothedSeries), [smoothedSeries]);

  const min = smoothedVals.length ? Math.min(...smoothedVals) : 0;
  const max = smoothedVals.length ? Math.max(...smoothedVals) : 0;
  const mid = (min + max) / 2;
  const xStart = 0;
  const xEnd = normalized.length ? normalized[normalized.length - 1].step : 0;
  const xMid = Math.round(xEnd / 2);

  const fmtAxis = (v) => {
    const n = Number(v || 0);
    if (Math.abs(n) < 0.001 && n !== 0) return n.toExponential(1);
    return n.toFixed(3);
  };

  return (
    <View style={styles.chartCard}>
      <Text style={styles.chartTitle}>{title}</Text>

      <View style={styles.chartBody}>
        <View style={styles.yAxisCol}>
          <Text style={styles.axisLabel}>{fmtAxis(max)}</Text>
          <Text style={styles.axisLabel}>{fmtAxis(mid)}</Text>
          <Text style={styles.axisLabel}>{fmtAxis(min)}</Text>
        </View>

        <View style={styles.plotCol}>
          <Svg width="100%" height="110" viewBox="0 0 320 110">
            <Line x1="0" y1="8" x2="320" y2="8" stroke="rgba(255,255,255,0.11)" strokeWidth="1" />
            <Line x1="0" y1="55" x2="320" y2="55" stroke="rgba(255,255,255,0.08)" strokeWidth="1" />
            <Line x1="0" y1="102" x2="320" y2="102" stroke="rgba(255,255,255,0.11)" strokeWidth="1" />
            <Polyline
              points={points}
              fill="none"
              stroke={color}
              strokeWidth="2.6"
              strokeLinejoin="round"
              strokeLinecap="round"
            />
          </Svg>
          <View style={styles.xAxisRow}>
            <Text style={styles.axisLabel}>step {xStart}</Text>
            <Text style={styles.axisLabel}>step {xMid}</Text>
            <Text style={styles.axisLabel}>step {xEnd}</Text>
          </View>
        </View>
      </View>

      <Text style={styles.hintText}>{hint}</Text>
    </View>
  );
}

export default function App() {
  const [data, setData] = useState(null);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState('');

  const fetchMetrics = useCallback(async () => {
    try {
      const r = await fetch(API_URL);
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const j = await r.json();
      setData(j);
      setError('');
    } catch (e) {
      setError(String(e.message || e));
    }
  }, []);

  useEffect(() => {
    fetchMetrics();
    const t = setInterval(fetchMetrics, REFRESH_MS);
    return () => clearInterval(t);
  }, [fetchMetrics]);

  const onRefresh = useCallback(async () => {
    setRefreshing(true);
    await fetchMetrics();
    setRefreshing(false);
  }, [fetchMetrics]);

  const summary = data?.summary || {};
  const trainLossSeries = data?.series?.loss || [];
  const evalLossSeries = data?.series?.evalLoss || [];
  const gradNormSeries = data?.series?.gradNorm || [];
  const lrSeries = data?.series?.lr || [];

  return (
    <SafeAreaView style={styles.safe}>
      <StatusBar barStyle="light-content" />
      <ScrollView
        style={styles.scroll}
        contentContainerStyle={styles.content}
        refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} tintColor="#ff8f3f" />}
      >
        <View style={styles.headerRow}>
          <View>
            <Text style={styles.title}>ColQwen Monster Ops</Text>
            <Text style={styles.subtitle}>iPhone mission console • CGI Monster v2 7K</Text>
          </View>
          <View style={styles.pill}>
            <Text style={styles.pillText}>{(data?.status || '...').toUpperCase()}</Text>
          </View>
        </View>

        {!!error && (
          <View style={[styles.banner, { borderColor: '#ff6a8d' }]}>
            <Text style={styles.bannerText}>Fetch error: {error}</Text>
            <Text style={styles.bannerHint}>Check dashboard server at {API_URL}</Text>
          </View>
        )}

        <View style={styles.kpiGrid}>
          <KPI
            label="Step"
            value={summary.latestStep ?? '-'}
            sub={`${(summary.progressPct ?? 0).toFixed?.(2) || 0}% progress`}
            accent="#ff8f3f"
          />
          <KPI
            label="Train Loss"
            value={summary.latestLoss != null ? Number(summary.latestLoss).toFixed(4) : '-'}
            sub="optimization pulse"
            accent="#ff4d6d"
          />
          <KPI
            label="Eval Loss"
            value={summary.latestEvalLoss != null ? Number(summary.latestEvalLoss).toFixed(4) : 'N/A'}
            sub={summary.latestEvalLoss != null ? 'holdout check' : 'not logged in this run'}
            accent="#ffdd57"
          />
          <KPI
            label="Grad Norm"
            value={summary.latestGradNorm != null ? Number(summary.latestGradNorm).toFixed(4) : 'N/A'}
            sub={summary.latestGradNorm != null ? 'update stability' : 'not logged in this run'}
            accent="#ff6e40"
          />
          <KPI
            label="Learning Rate"
            value={summary.latestLr != null ? Number(summary.latestLr).toFixed(6) : 'N/A'}
            sub="optimizer step size"
            accent="#ffa94d"
          />
          <KPI
            label="Checkpoints"
            value={summary.checkpointCount ?? '-'}
            sub={`latest @ ${summary.latestCheckpoint?.step ?? '-'}`}
            accent="#ffc078"
          />
        </View>

        <SparkCard
          title="Train Loss"
          series={trainLossSeries}
          color="#ff4d6d"
          hint="Core optimization objective. Trend down = retrieval head learning better alignment."
        />

        <SparkCard
          title="Eval Loss"
          series={evalLossSeries}
          color="#ffdd57"
          hint="Holdout generalization signal. Flat/down is good; sharp divergence vs train means overfit risk."
        />

        <SparkCard
          title="Grad Norm"
          series={gradNormSeries}
          color="#ff6e40"
          hint="Update magnitude stability. Spikes suggest turbulence; too-flat may indicate learning stall."
        />

        <SparkCard
          title="Learning Rate"
          series={lrSeries}
          color="#ff8f3f"
          hint="Optimizer step size schedule over time. Useful when warmup/decay are enabled."
        />

        <View style={styles.gpuCard}>
          <Text style={styles.chartTitle}>GPU Runtime Snapshot</Text>
          {(data?.gpus || []).map((g) => {
            const util = Number(g.util || 0);
            const memPct = g.memTotal ? Math.round((Number(g.memUsed || 0) / Number(g.memTotal || 1)) * 100) : 0;
            return (
              <View key={g.index} style={styles.gpuBox}>
                <Text style={styles.gpuTitle}>GPU {g.index}</Text>
                <Text style={styles.gpuMeta}>util {util}% • mem {g.memUsed}/{g.memTotal} MB • {g.temp}°C • {Number(g.power).toFixed(1)}W</Text>
                <View style={styles.meterTrack}>
                  <View style={[styles.meterBar, { width: `${Math.max(util, memPct)}%` }]} />
                </View>
              </View>
            );
          })}
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: '#140606' },
  scroll: { flex: 1 },
  content: {
    paddingHorizontal: 12,
    paddingTop: 8,
    paddingBottom: 24,
    gap: 10,
  },
  headerRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-end',
    marginBottom: 2,
  },
  title: {
    color: '#ff8f3f',
    fontSize: 22,
    fontWeight: '800',
  },
  subtitle: { color: '#a4abd8', fontSize: 11, marginTop: 2 },
  pill: {
    borderColor: 'rgba(70,255,216,0.45)',
    borderWidth: 1,
    borderRadius: 999,
    paddingHorizontal: 10,
    paddingVertical: 5,
    backgroundColor: 'rgba(70,255,216,0.08)',
  },
  pillText: { color: '#dffeff', fontSize: 11, fontWeight: '700' },
  banner: {
    borderWidth: 1,
    borderRadius: 10,
    padding: 8,
    backgroundColor: 'rgba(255,90,130,0.12)',
  },
  bannerText: { color: '#ffd6e1', fontSize: 12, fontWeight: '700' },
  bannerHint: { color: '#ffbdd0', fontSize: 10, marginTop: 2 },
  kpiGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    marginHorizontal: -4,
  },
  kpiCard: {
    width: '50%',
    paddingHorizontal: 4,
    paddingVertical: 4,
  },
  kpiLabel: {
    color: '#9ba4d7',
    fontSize: 10,
    textTransform: 'uppercase',
    letterSpacing: 0.8,
  },
  kpiValue: {
    fontSize: 16,
    fontWeight: '800',
    marginTop: 2,
  },
  kpiSub: {
    color: '#8b93c2',
    fontSize: 10,
    marginTop: 2,
  },
  chartCard: {
    borderWidth: 1,
    borderColor: 'rgba(70,255,216,0.2)',
    borderRadius: 12,
    backgroundColor: 'rgba(18,8,40,0.78)',
    paddingHorizontal: 10,
    paddingTop: 8,
    paddingBottom: 9,
  },
  chartTitle: {
    color: '#e8ecff',
    fontWeight: '700',
    fontSize: 12,
    marginBottom: 2,
  },
  hintText: {
    color: '#9ea5d3',
    fontSize: 10,
    lineHeight: 13,
  },
  chartBody: {
    flexDirection: 'row',
    alignItems: 'stretch',
    marginBottom: 2,
  },
  yAxisCol: {
    width: 46,
    justifyContent: 'space-between',
    paddingRight: 4,
    paddingTop: 2,
    paddingBottom: 6,
  },
  plotCol: {
    flex: 1,
  },
  xAxisRow: {
    marginTop: -2,
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingHorizontal: 2,
  },
  axisLabel: {
    color: '#7f88bc',
    fontSize: 9,
    fontVariant: ['tabular-nums'],
  },
  gpuCard: {
    borderWidth: 1,
    borderColor: 'rgba(255,214,106,0.23)',
    borderRadius: 12,
    backgroundColor: 'rgba(20,10,44,0.82)',
    padding: 10,
  },
  gpuBox: {
    marginTop: 8,
    padding: 8,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.14)',
    backgroundColor: 'rgba(7,10,27,0.68)',
  },
  gpuTitle: { color: '#d9fffa', fontWeight: '700', fontSize: 12 },
  gpuMeta: { color: '#a3abda', fontSize: 10, marginTop: 2 },
  meterTrack: {
    height: 7,
    borderRadius: 999,
    backgroundColor: 'rgba(255,255,255,0.09)',
    marginTop: 6,
    overflow: 'hidden',
  },
  meterBar: {
    height: '100%',
    backgroundColor: '#ff8f3f',
  },
});
