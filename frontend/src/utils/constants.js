export const API_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

export const LABEL_INFO = {
  "pants-fire": {
    title: "Pants on Fire",
    description: "LIAR dataset benchmark classification (Blatantly False)",
    level: 1,
    color: "#b91c1c",
    bg: "#fef2f2",
    border: "#fecaca",
  },
  false: {
    title: "False",
    description: "LIAR dataset benchmark classification (False Statement)",
    level: 2,
    color: "#dc2626",
    bg: "#fef2f2",
    border: "#fecaca",
  },
  "barely-true": {
    title: "Barely True",
    description: "LIAR dataset benchmark classification (Mostly Untrue)",
    level: 3,
    color: "#d97706",
    bg: "#fffbeb",
    border: "#fde68a",
  },
  "half-true": {
    title: "Half True",
    description: "LIAR dataset benchmark classification (Partially Accurate)",
    level: 4,
    color: "#d97706",
    bg: "#fffbeb",
    border: "#fde68a",
  },
  "mostly-true": {
    title: "Mostly True",
    description: "LIAR dataset benchmark classification (Substantially Accurate)",
    level: 5,
    color: "#0284c7",
    bg: "#f0f9ff",
    border: "#bae6fd",
  },
  true: {
    title: "True",
    description: "LIAR dataset benchmark classification (Factually Accurate)",
    level: 6,
    color: "#16a34a",
    bg: "#f0fdf4",
    border: "#bbf7d0",
  },
};

export const EXAMPLES = [
  "The state unemployment rate dropped below 5 percent following new manufacturing investments.",
  "Federal healthcare spending increased by 20 percent under the recent budget bill.",
  "The proposed tax reform will eliminate deductions for middle-income households.",
  "BREAKING: First NFL Team Declares Bankruptcy Over Kneeling Thugs",
  "Clean energy subsidies created thousands of domestic solar and wind manufacturing jobs.",
];

export function formatLabel(label) {
  if (!label) return "Unknown";
  if (label === "pants-fire") return "Pants on Fire";
  return label
    .split("-")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
}

export function formatSpeaker(speaker) {
  if (!speaker) return "Not specified";
  return speaker
    .split("-")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
}

export function formatAffiliation(party) {
  if (!party || party.toLowerCase() === "none") return "";
  return party.charAt(0).toUpperCase() + party.slice(1);
}

export function formatDateTime(isoString) {
  if (!isoString) return "N/A";
  try {
    const d = new Date(isoString);
    if (isNaN(d.getTime())) return isoString;
    return d.toLocaleString(undefined, {
      year: "numeric",
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
      timeZoneName: "short",
    });
  } catch {
    return isoString;
  }
}

export function getLabelInfo(label) {
  return (
    LABEL_INFO[label] || {
      title: formatLabel(label),
      description: "LIAR six-class classification",
      level: 0,
      color: "#64748b",
      bg: "#f1f5f9",
      border: "#e2e8f0",
    }
  );
}
