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
import Svg, { Polyline } from 'react-native-svg';

const API_URL = 'http://10.46.150.108:3472/api/metrics';
const REFRESH_MS = 5000;

function sample(points = [], maxPoints = 42) {
  if (!points?.length || points.length <= maxPoints) return points;
  const stride = Math.ceil(points.length / maxPoints);
  return points.filter((_, i) => i % stride === 0 || i === points.length - 1);
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

function toPolylinePoints(values = [], width = 320, height = 110, pad = 6) {
  if (!values.length) return '';
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = Math.max(max - min, 1e-6);
  return values
    .map((v, i) => {
      const x = pad + (i / Math.max(values.length - 1, 1)) * (width - pad * 2);
      const y = height - pad - ((v - min) / span) * (height - pad * 2);
      return `${x},${y}`;
    })
    .join(' ');
}

function KPI({ label, value, sub, accent = '#46ffd8' }) {
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

function SparkCard({ title, hint, values, color = '#46ffd8' }) {
  const sampled = useMemo(() => smooth(sample(values, 48)), [values]);
  const points = useMemo(() => toPolylinePoints(sampled), [sampled]);

  return (
    <View style={styles.chartCard}>
      <Text style={styles.chartTitle}>{title}</Text>
      <Svg width="100%" height="110" viewBox="0 0 320 110">
        <Polyline
          points={points}
          fill="none"
          stroke={color}
          strokeWidth="2.6"
          strokeLinejoin="round"
          strokeLinecap="round"
        />
      </Svg>
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
  const rolling10 = (data?.series?.rolling10 || []).map((x) => x.loss);
  const rolling50 = (data?.series?.rolling50 || []).map((x) => x.loss);
  const rawLoss = (data?.series?.loss || []).map((x) => x.loss);

  return (
    <SafeAreaView style={styles.safe}>
      <StatusBar barStyle="light-content" />
      <ScrollView
        style={styles.scroll}
        contentContainerStyle={styles.content}
        refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} tintColor="#46ffd8" />}
      >
        <View style={styles.headerRow}>
          <View>
            <Text style={styles.title}>GemmPali Neon Ops</Text>
            <Text style={styles.subtitle}>iPhone mission console • multi-page dominance</Text>
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
            accent="#46ffd8"
          />
          <KPI
            label="Loss"
            value={summary.latestLoss != null ? Number(summary.latestLoss).toFixed(4) : '-'}
            sub="current pulse"
            accent="#ff62d8"
          />
          <KPI
            label="Checkpoints"
            value={summary.checkpointCount ?? '-'}
            sub={`latest @ ${summary.latestCheckpoint?.step ?? '-'}`}
            accent="#ffd86a"
          />
          <KPI
            label="Spikes"
            value={summary.spikeCount ?? '-'}
            sub="loss > 0.10"
            accent="#b489ff"
          />
        </View>

        <SparkCard
          title="Trend Short (Rolling-10)"
          values={rolling10}
          color="#46ffd8"
          hint="Near-term drift. Quick bumps are normal under hard negatives."
        />

        <SparkCard
          title="Trend Long (Rolling-50)"
          values={rolling50}
          color="#ffd86a"
          hint="Structural trend. This is your truth line for stability."
        />

        <SparkCard
          title="Raw Loss (Hidden Noise Layer)"
          values={rawLoss}
          color="#ff62d8"
          hint="Noisy by design; use for anomaly detection, not macro judgment."
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
  safe: { flex: 1, backgroundColor: '#05010d' },
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
    color: '#46ffd8',
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
    fontSize: 18,
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
    backgroundColor: '#46ffd8',
  },
});
