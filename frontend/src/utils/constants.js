export const API_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

export const TARGET_DOMAIN = "Socio-Economic & Domestic Public Welfare Policy";
export const DOMAIN_CLASSES_3 = ["Misleading", "Half-True", "Factual"];

export const LABEL_INFO = {
  // Socio-Economic 3-Class Approved Veracity Labels
  misleading: {
    title: "Misleading",
    description: "Socio-Economic domain benchmark classification (Substantively False / Fabricated)",
    level: 2,
    color: "#dc2626",
    bg: "#fef2f2",
    border: "#fecaca",
  },
  "half-true": {
    title: "Half-True",
    description: "Socio-Economic domain benchmark classification (Partially Accurate with Omitted Context)",
    level: 4,
    color: "#d97706",
    bg: "#fffbeb",
    border: "#fde68a",
  },
  factual: {
    title: "Factual",
    description: "Socio-Economic domain benchmark classification (Substantially Accurate & Corroborated)",
    level: 6,
    color: "#16a34a",
    bg: "#f0fdf4",
    border: "#bbf7d0",
  },

  // LIAR 6-Class Benchmark Veracity Labels
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
  "Under the Affordable Care Act, the national uninsured rate dropped below 9 percent while hospital uncompensated care costs declined significantly.",
  "The state unemployment rate dropped below 4.5 percent following new manufacturing investments and statewide workforce development initiatives.",
  "The proposed federal tax reform will eliminate standard deductions and raise effective tax rates on middle-income working families.",
  "State budget appropriations for public K-12 education were reduced by over $400 million, lowering per-pupil classroom spending.",
  "Over 70 percent of Supplemental Nutrition Assistance Program (SNAP) benefits are distributed to households with children, seniors, or disabled individuals.",
  "The expanded Child Tax Credit reduced nationwide child poverty by roughly 50 percent during its first full year of implementation.",
  "The federal minimum wage adjusted for inflation is worth approximately 20 percent less than it was forty years ago.",
  "Medicare and Medicaid expenditures currently account for nearly half of all mandatory federal budget outlays.",
];

export function formatLabel(label) {
  if (!label) return "Unknown";
  const clean = String(label).trim();
  const lower = clean.toLowerCase();
  if (lower === "pants-fire") return "Pants on Fire";
  if (lower === "misleading") return "Misleading";
  if (lower === "half-true") return "Half-True";
  if (lower === "factual") return "Factual";
  return clean
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
  if (!label) {
    return {
      title: "Unknown",
      description: "Unspecified classification",
      level: 0,
      color: "#64748b",
      bg: "#f1f5f9",
      border: "#e2e8f0",
    };
  }
  const clean = String(label).trim().toLowerCase();
  return (
    LABEL_INFO[clean] ||
    LABEL_INFO[label] || {
      title: formatLabel(label),
      description: "Classification",
      level: 0,
      color: "#64748b",
      bg: "#f1f5f9",
      border: "#e2e8f0",
    }
  );
}
