import { useEffect, useMemo, useRef, useState } from "react";
import ForceGraph2D from "react-force-graph-2d";
import {
  Activity,
  AlertTriangle,
  ArrowRight,
  BadgeDollarSign,
  CircleAlert,
  Gauge,
  Loader2,
  Network,
  RefreshCcw,
  Search,
  ShieldCheck,
  ShieldX,
  WifiOff,
} from "lucide-react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  API_BASE_URL,
  getApiErrorMessage,
  getDriftStatus,
  getExplanation,
  getHealth,
  getPrediction,
} from "../services/api";

const INITIAL_ACCOUNT_ID = "19204";

function classNames(...values) {
  return values.filter(Boolean).join(" ");
}

function asPercent(value) {
  if (typeof value !== "number" || Number.isNaN(value)) {
    return "0.0%";
  }

  return `${(value * 100).toFixed(1)}%`;
}

function formatNumber(value, options = {}) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "-";
  }

  return new Intl.NumberFormat("en-US", options).format(Number(value));
}

function riskBand(prediction) {
  if (!prediction) {
    return { label: "PENDING", tone: "neutral", color: "#64748b" };
  }

  if (prediction.fraud_probability >= 0.85) {
    return { label: "CRITICAL", tone: "danger", color: "#e11d48" };
  }

  if (prediction.is_high_risk) {
    return { label: "HIGH RISK", tone: "warning", color: "#f59e0b" };
  }

  return { label: "BENIGN", tone: "success", color: "#14b8a6" };
}

function StatusPill({ label, tone = "neutral", icon: Icon }) {
  const styles = {
    danger: "border-rose-200 bg-rose-50 text-rose-700",
    warning: "border-amber-200 bg-amber-50 text-amber-700",
    success: "border-teal-200 bg-teal-50 text-teal-700",
    neutral: "border-slate-200 bg-slate-50 text-slate-600",
    info: "border-indigo-200 bg-indigo-50 text-indigo-700",
  };

  return (
    <span
      className={classNames(
        "inline-flex min-h-8 items-center gap-2 rounded-md border px-2.5 py-1 text-xs font-semibold uppercase tracking-normal",
        styles[tone],
      )}
    >
      {Icon ? <Icon className={classNames("h-3.5 w-3.5", Icon === Loader2 && "animate-spin")} aria-hidden="true" /> : null}
      {label}
    </span>
  );
}

function RiskRing({ prediction }) {
  const band = riskBand(prediction);
  const value = Math.max(0, Math.min(1, prediction?.fraud_probability ?? 0));

  return (
    <div
      className="risk-ring relative"
      style={{ "--pct": value * 100, "--risk-color": band.color }}
      aria-label={`Fraud probability ${asPercent(value)}`}
    >
      <div className="risk-ring__content">
        <span className="text-3xl font-bold text-slate-950">{asPercent(value)}</span>
        <span className="text-xs font-semibold uppercase text-slate-500">Risk Score</span>
      </div>
    </div>
  );
}

function MetricTile({ icon: Icon, label, value, tone = "neutral" }) {
  const iconStyles = {
    danger: "bg-rose-100 text-rose-700",
    warning: "bg-amber-100 text-amber-700",
    success: "bg-teal-100 text-teal-700",
    neutral: "bg-slate-100 text-slate-700",
    info: "bg-indigo-100 text-indigo-700",
  };

  return (
    <div className="flex items-center gap-3 rounded-md bg-slate-50 p-3">
      <span className={classNames("grid h-10 w-10 shrink-0 place-items-center rounded-md", iconStyles[tone])}>
        <Icon className="h-5 w-5" aria-hidden="true" />
      </span>
      <div className="min-w-0">
        <p className="text-xs font-semibold uppercase text-slate-500">{label}</p>
        <p className="truncate text-lg font-bold text-slate-950">{value}</p>
      </div>
    </div>
  );
}

function EmptyState({ icon: Icon, title, message }) {
  return (
    <div className="grid min-h-56 place-items-center rounded-lg border border-dashed border-slate-300 bg-white p-6 text-center">
      <div>
        <Icon className="mx-auto h-9 w-9 text-slate-400" aria-hidden="true" />
        <p className="mt-3 text-sm font-semibold text-slate-900">{title}</p>
        <p className="mt-1 max-w-md text-sm text-slate-500">{message}</p>
      </div>
    </div>
  );
}

function FeatureChart({ topFeatures }) {
  const data = useMemo(
    () =>
      Object.entries(topFeatures ?? {})
        .map(([name, value]) => ({
          name,
          value: Number(value),
          label: name.replaceAll("_", " "),
        }))
        .sort((a, b) => b.value - a.value),
    [topFeatures],
  );

  if (!data.length) {
    return <EmptyState icon={Gauge} title="No feature weights" message="Explanation data will appear after the account is analyzed." />;
  }

  return (
    <div className="h-80 min-h-80">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} layout="vertical" margin={{ left: 24, right: 24, top: 8, bottom: 8 }}>
          <CartesianGrid horizontal={false} stroke="#e2e8f0" />
          <XAxis type="number" domain={[0, "dataMax"]} tick={{ fill: "#64748b", fontSize: 12 }} />
          <YAxis
            type="category"
            dataKey="label"
            tick={{ fill: "#334155", fontSize: 12 }}
            width={150}
            interval={0}
          />
          <Tooltip
            cursor={{ fill: "rgba(99, 102, 241, 0.08)" }}
            formatter={(value) => [Number(value).toFixed(4), "Importance"]}
          />
          <Bar dataKey="value" fill="#4f46e5" radius={[0, 5, 5, 0]} maxBarSize={24} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

function TransactionGraph({ transactions, accountId }) {
  const containerRef = useRef(null);
  const [width, setWidth] = useState(520);

  useEffect(() => {
    if (!containerRef.current) {
      return undefined;
    }

    const observer = new ResizeObserver(([entry]) => {
      setWidth(Math.max(280, Math.floor(entry.contentRect.width)));
    });

    observer.observe(containerRef.current);
    return () => observer.disconnect();
  }, []);

  const graphData = useMemo(() => {
    const nodesById = new Map();
    const links = [];
    const rootId = String(accountId || INITIAL_ACCOUNT_ID);

    nodesById.set(rootId, { id: rootId, label: `Account ${rootId}`, group: "subject" });

    transactions.forEach((transaction, index) => {
      const source = String(transaction.source_node_id ?? transaction.source_index ?? rootId);
      const target = String(transaction.target_node_id ?? transaction.target_index ?? `counterparty-${index + 1}`);
      const transactionId = String(transaction.transaction_id ?? `transaction-${index + 1}`);

      nodesById.set(source, { id: source, label: `Account ${source}`, group: source === rootId ? "subject" : "counterparty" });
      nodesById.set(target, { id: target, label: `Account ${target}`, group: target === rootId ? "subject" : "counterparty" });
      links.push({
        source,
        target,
        transactionId,
        amount: Number(transaction.amount ?? 0),
        importance: Number(transaction.importance ?? 0),
      });
    });

    return {
      nodes: Array.from(nodesById.values()),
      links,
    };
  }, [accountId, transactions]);

  if (!transactions.length) {
    return <EmptyState icon={Network} title="No critical edges" message="Critical transaction links will appear after explanation data is returned." />;
  }

  return (
    <div ref={containerRef} className="h-80 overflow-hidden rounded-lg border border-slate-200 bg-slate-950">
      <ForceGraph2D
        graphData={graphData}
        width={width}
        height={320}
        backgroundColor="#020617"
        nodeLabel="label"
        linkLabel={(link) => `Transaction ${link.transactionId}: $${formatNumber(link.amount, { maximumFractionDigits: 2 })}`}
        linkColor={() => "rgba(251, 191, 36, 0.78)"}
        linkDirectionalArrowLength={5}
        linkDirectionalArrowRelPos={0.92}
        linkWidth={(link) => Math.max(1.5, link.importance * 5)}
        nodeCanvasObject={(node, ctx, globalScale) => {
          const label = node.id;
          const radius = node.group === "subject" ? 7 : 5;
          const fontSize = Math.max(9, 12 / globalScale);

          ctx.beginPath();
          ctx.arc(node.x, node.y, radius, 0, 2 * Math.PI, false);
          ctx.fillStyle = node.group === "subject" ? "#14b8a6" : "#94a3b8";
          ctx.fill();
          ctx.strokeStyle = "#ffffff";
          ctx.lineWidth = 1.2;
          ctx.stroke();
          ctx.font = `${fontSize}px Inter, sans-serif`;
          ctx.fillStyle = "#e2e8f0";
          ctx.textAlign = "center";
          ctx.fillText(label, node.x, node.y + radius + fontSize + 2);
        }}
      />
    </div>
  );
}

function TransactionsTable({ transactions }) {
  if (!transactions.length) {
    return <EmptyState icon={BadgeDollarSign} title="No transaction rows" message="The audited transaction table is waiting for explanation output." />;
  }

  return (
    <div className="overflow-hidden rounded-lg border border-slate-200">
      <div className="thin-scrollbar max-h-80 overflow-auto">
        <table className="min-w-full divide-y divide-slate-200 text-sm">
          <thead className="sticky top-0 bg-slate-50 text-left text-xs font-semibold uppercase text-slate-500">
            <tr>
              <th className="px-4 py-3">Transaction</th>
              <th className="px-4 py-3">Route</th>
              <th className="px-4 py-3 text-right">Amount</th>
              <th className="px-4 py-3 text-right">Event Time</th>
              <th className="px-4 py-3 text-right">Importance</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100 bg-white">
            {transactions.map((transaction, index) => (
              <tr key={`${transaction.transaction_id ?? "tx"}-${index}`}>
                <td className="whitespace-nowrap px-4 py-3 font-medium text-slate-900">
                  {transaction.transaction_id ?? `Edge ${index + 1}`}
                </td>
                <td className="px-4 py-3 text-slate-600">
                  <div className="flex min-w-52 items-center gap-2">
                    <span>{transaction.source_node_id ?? transaction.source_index ?? "-"}</span>
                    <ArrowRight className="h-3.5 w-3.5 shrink-0 text-slate-400" aria-hidden="true" />
                    <span>{transaction.target_node_id ?? transaction.target_index ?? "-"}</span>
                  </div>
                </td>
                <td className="whitespace-nowrap px-4 py-3 text-right text-slate-700">
                  ${formatNumber(transaction.amount, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                </td>
                <td className="whitespace-nowrap px-4 py-3 text-right text-slate-700">
                  {formatNumber(transaction.event_time, { maximumFractionDigits: 2 })}
                </td>
                <td className="whitespace-nowrap px-4 py-3 text-right text-slate-700">
                  {formatNumber(transaction.importance, { maximumFractionDigits: 4 })}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function AlertQueue({ accounts, activeAccountId, onSelect }) {
  if (!accounts.length) {
    return (
      <div className="rounded-lg border border-dashed border-slate-300 bg-slate-50 p-4 text-sm text-slate-500">
        Account results will be listed here.
      </div>
    );
  }

  return (
    <div className="space-y-2">
      {accounts.map((account) => {
        const band = riskBand(account);
        return (
          <button
            key={account.account_id}
            type="button"
            onClick={() => onSelect(String(account.account_id))}
            className={classNames(
              "w-full rounded-lg border p-3 text-left transition hover:border-slate-400 hover:bg-slate-50",
              String(activeAccountId) === String(account.account_id)
                ? "border-indigo-300 bg-indigo-50"
                : "border-slate-200 bg-white",
            )}
          >
            <div className="flex items-center justify-between gap-3">
              <span className="font-semibold text-slate-950">Account {account.account_id}</span>
              <StatusPill label={band.label} tone={band.tone} />
            </div>
            <div className="mt-3 flex items-center justify-between text-sm text-slate-500">
              <span>Probability</span>
              <span className="font-semibold text-slate-800">{asPercent(account.fraud_probability)}</span>
            </div>
          </button>
        );
      })}
    </div>
  );
}

export default function ForensicDashboard() {
  const [accountId, setAccountId] = useState(INITIAL_ACCOUNT_ID);
  const [prediction, setPrediction] = useState(null);
  const [explanation, setExplanation] = useState(null);
  const [health, setHealth] = useState(null);
  const [drift, setDrift] = useState(null);
  const [alertQueue, setAlertQueue] = useState([]);
  const [investigationError, setInvestigationError] = useState("");
  const [opsError, setOpsError] = useState("");
  const [isInvestigating, setIsInvestigating] = useState(false);
  const [isRefreshingOps, setIsRefreshingOps] = useState(false);

  const criticalTransactions = explanation?.critical_transactions ?? [];
  const band = riskBand(prediction);

  async function refreshOperations() {
    setIsRefreshingOps(true);
    setOpsError("");

    try {
      const [healthPayload, driftPayload] = await Promise.all([getHealth(), getDriftStatus()]);
      setHealth(healthPayload);
      setDrift(driftPayload);
    } catch (error) {
      setOpsError(getApiErrorMessage(error));
    } finally {
      setIsRefreshingOps(false);
    }
  }

  async function investigateAccount(targetAccountId = accountId) {
    const normalizedAccountId = String(targetAccountId).trim();

    if (!normalizedAccountId) {
      setInvestigationError("Account ID is required.");
      return;
    }

    setAccountId(normalizedAccountId);
    setIsInvestigating(true);
    setInvestigationError("");

    try {
      const [predictionPayload, explanationPayload] = await Promise.all([
        getPrediction(normalizedAccountId),
        getExplanation(normalizedAccountId),
      ]);

      setPrediction(predictionPayload);
      setExplanation(explanationPayload);
      setAlertQueue((current) => {
        const merged = current.filter((account) => String(account.account_id) !== String(predictionPayload.account_id));
        return [predictionPayload, ...merged]
          .sort((a, b) => b.fraud_probability - a.fraud_probability)
          .slice(0, 8);
      });
    } catch (error) {
      setPrediction(null);
      setExplanation(null);
      setInvestigationError(getApiErrorMessage(error));
    } finally {
      setIsInvestigating(false);
    }
  }

  useEffect(() => {
    refreshOperations();
    investigateAccount(INITIAL_ACCOUNT_ID);
  }, []);

  return (
    <main className="min-h-screen p-4 text-slate-900 sm:p-6 lg:p-8">
      <div className="mx-auto flex max-w-7xl flex-col gap-5">
        <header className="flex flex-col gap-4 rounded-lg border border-slate-200 bg-white p-5 shadow-panel lg:flex-row lg:items-center lg:justify-between">
          <div className="min-w-0">
            <p className="text-sm font-semibold uppercase text-teal-700">Financial Risk Intelligence</p>
            <h1 className="mt-1 text-2xl font-bold text-slate-950 sm:text-3xl">Analyst Console</h1>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <StatusPill
              label={health?.status ? `API ${health.status}` : "API Pending"}
              tone={health?.status === "healthy" ? "success" : opsError ? "danger" : "neutral"}
              icon={health?.status === "healthy" ? ShieldCheck : opsError ? WifiOff : Activity}
            />
            <StatusPill
              label={drift?.drift_detected ? "Drift Detected" : drift ? "Drift Stable" : "Drift Pending"}
              tone={drift?.drift_detected ? "warning" : drift ? "success" : "neutral"}
              icon={drift?.drift_detected ? AlertTriangle : ShieldCheck}
            />
            <button
              type="button"
              onClick={refreshOperations}
              className="inline-flex h-9 items-center gap-2 rounded-md border border-slate-200 bg-white px-3 text-sm font-semibold text-slate-700 shadow-sm transition hover:border-slate-400"
              title="Refresh operations"
            >
              <RefreshCcw className={classNames("h-4 w-4", isRefreshingOps && "animate-spin")} aria-hidden="true" />
              Refresh
            </button>
          </div>
        </header>

        {opsError ? (
          <div className="flex items-start gap-3 rounded-lg border border-rose-200 bg-rose-50 p-4 text-sm text-rose-700">
            <WifiOff className="mt-0.5 h-5 w-5 shrink-0" aria-hidden="true" />
            <span>{opsError}</span>
          </div>
        ) : null}

        <section className="grid gap-5 lg:grid-cols-[22rem_minmax(0,1fr)]">
          <aside className="space-y-5">
            <div className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
              <form
                className="space-y-3"
                onSubmit={(event) => {
                  event.preventDefault();
                  investigateAccount();
                }}
              >
                <label htmlFor="account-id" className="text-sm font-semibold text-slate-700">
                  Account ID
                </label>
                <div className="flex gap-2">
                  <input
                    id="account-id"
                    value={accountId}
                    inputMode="numeric"
                    onChange={(event) => setAccountId(event.target.value)}
                    className="min-w-0 flex-1 rounded-md border border-slate-300 bg-white px-3 py-2 text-base text-slate-950 shadow-sm"
                  />
                  <button
                    type="submit"
                    className="inline-flex h-11 w-11 shrink-0 items-center justify-center rounded-md bg-slate-950 text-white transition hover:bg-slate-800"
                    title="Analyze account"
                    disabled={isInvestigating}
                  >
                    {isInvestigating ? (
                      <Loader2 className="h-5 w-5 animate-spin" aria-hidden="true" />
                    ) : (
                      <Search className="h-5 w-5" aria-hidden="true" />
                    )}
                  </button>
                </div>
              </form>
            </div>

            <div className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
              <div className="mb-4 flex items-center justify-between">
                <h2 className="text-sm font-bold uppercase text-slate-700">Alert Queue</h2>
                <StatusPill label={`${alertQueue.length} Accounts`} tone="info" />
              </div>
              <AlertQueue accounts={alertQueue} activeAccountId={accountId} onSelect={investigateAccount} />
            </div>

            <div className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
              <h2 className="text-sm font-bold uppercase text-slate-700">MLOps Control</h2>
              <div className="mt-4 space-y-3">
                <MetricTile
                  icon={Activity}
                  label="Model"
                  value={health?.model ?? "-"}
                  tone={health?.status === "healthy" ? "success" : "neutral"}
                />
                <MetricTile
                  icon={Gauge}
                  label="Drift Score"
                  value={drift ? formatNumber(drift.drift_score, { maximumFractionDigits: 4 }) : "-"}
                  tone={drift?.drift_detected ? "warning" : "info"}
                />
                <MetricTile
                  icon={CircleAlert}
                  label="Drifted Features"
                  value={drift?.drifted_features?.length ?? "-"}
                  tone={drift?.drift_detected ? "warning" : "neutral"}
                />
              </div>
              {drift?.drifted_features?.length ? (
                <div className="mt-4 flex flex-wrap gap-2">
                  {drift.drifted_features.map((feature) => (
                    <StatusPill key={feature} label={feature.replaceAll("_", " ")} tone="warning" />
                  ))}
                </div>
              ) : null}
            </div>
          </aside>

          <div className="space-y-5">
            {investigationError ? (
              <div className="flex items-start gap-3 rounded-lg border border-rose-200 bg-rose-50 p-4 text-sm text-rose-700">
                <ShieldX className="mt-0.5 h-5 w-5 shrink-0" aria-hidden="true" />
                <span>{investigationError}</span>
              </div>
            ) : null}

            <section className="grid gap-5 xl:grid-cols-[20rem_minmax(0,1fr)]">
              <div className="rounded-lg border border-slate-200 bg-white p-5 shadow-sm">
                <div className="flex items-center justify-between gap-3">
                  <div>
                    <p className="text-sm font-semibold uppercase text-slate-500">Account</p>
                    <h2 className="mt-1 text-2xl font-bold text-slate-950">{prediction?.account_id ?? accountId}</h2>
                  </div>
                  <StatusPill label={band.label} tone={band.tone} />
                </div>
                <div className="mt-8 flex justify-center">
                  <RiskRing prediction={prediction} />
                </div>
                <div className="mt-8 grid grid-cols-2 gap-3">
                  <MetricTile
                    icon={Gauge}
                    label="Threshold"
                    value={prediction ? formatNumber(prediction.threshold_used, { maximumFractionDigits: 2 }) : "-"}
                    tone="info"
                  />
                  <MetricTile
                    icon={prediction?.is_high_risk ? ShieldX : ShieldCheck}
                    label="Decision"
                    value={prediction ? (prediction.is_high_risk ? "Review" : "Clear") : "-"}
                    tone={prediction?.is_high_risk ? "danger" : "success"}
                  />
                </div>
              </div>

              <div className="rounded-lg border border-slate-200 bg-white p-5 shadow-sm">
                <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
                  <div>
                    <p className="text-sm font-semibold uppercase text-slate-500">GNNExplainer</p>
                    <h2 className="mt-1 text-xl font-bold text-slate-950">Top Feature Weights</h2>
                  </div>
                  {isInvestigating ? <StatusPill label="Loading" tone="info" icon={Loader2} /> : null}
                </div>
                <FeatureChart topFeatures={explanation?.top_features} />
              </div>
            </section>

            <section className="grid gap-5 xl:grid-cols-2">
              <div className="rounded-lg border border-slate-200 bg-white p-5 shadow-sm">
                <div className="mb-4 flex items-center justify-between gap-3">
                  <div>
                    <p className="text-sm font-semibold uppercase text-slate-500">Edge Attribution</p>
                    <h2 className="mt-1 text-xl font-bold text-slate-950">Transaction Network</h2>
                  </div>
                  <StatusPill label={`${criticalTransactions.length} Edges`} tone="info" icon={Network} />
                </div>
                <TransactionGraph transactions={criticalTransactions} accountId={prediction?.account_id ?? accountId} />
              </div>

              <div className="rounded-lg border border-slate-200 bg-white p-5 shadow-sm">
                <div className="mb-4 flex items-center justify-between gap-3">
                  <div>
                    <p className="text-sm font-semibold uppercase text-slate-500">Audit Trail</p>
                    <h2 className="mt-1 text-xl font-bold text-slate-950">Critical Transactions</h2>
                  </div>
                  <StatusPill label={API_BASE_URL.replace(/^https?:\/\//, "")} tone="neutral" />
                </div>
                <TransactionsTable transactions={criticalTransactions} />
              </div>
            </section>
          </div>
        </section>
      </div>
    </main>
  );
}
