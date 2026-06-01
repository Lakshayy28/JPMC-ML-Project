import axios from "axios";

export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

const client = axios.create({
  baseURL: API_BASE_URL,
  timeout: 20000,
  headers: {
    "Content-Type": "application/json",
  },
});

export const DRIFT_SAMPLE = [
  { outgoing_tx_velocity_30d: 12.0, total_amount: 5000.0 },
  { outgoing_tx_velocity_30d: 11.4, total_amount: 4825.0 },
  { outgoing_tx_velocity_30d: 12.8, total_amount: 5210.0 },
  { outgoing_tx_velocity_30d: 10.9, total_amount: 4760.0 },
  { outgoing_tx_velocity_30d: 13.1, total_amount: 5480.0 },
  { outgoing_tx_velocity_30d: 11.8, total_amount: 5065.0 },
];

export async function getHealth() {
  const response = await client.get("/health");
  return response.data;
}

export async function getPrediction(accountId) {
  const response = await client.get(`/predict/${encodeURIComponent(accountId)}`);
  return response.data;
}

export async function getExplanation(accountId, epochs = 12) {
  const response = await client.get(`/explain/${encodeURIComponent(accountId)}`, {
    params: { epochs },
    timeout: 45000,
  });
  return response.data;
}

export async function getDriftStatus(recentFeatures = DRIFT_SAMPLE) {
  const response = await client.post("/analyze-drift", recentFeatures);
  return response.data;
}

export function getApiErrorMessage(error) {
  if (error.response?.data?.detail) {
    return error.response.data.detail;
  }

  if (error.code === "ECONNABORTED") {
    return "The API request timed out.";
  }

  if (error.message) {
    return error.message;
  }

  return "The API request failed.";
}
