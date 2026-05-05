export function yesNoLabel(value) {
  return value ? "Yes" : "No";
}

export function titleCase(value) {
  return String(value)
    .split("_")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

export function blendColor(start, end, ratio) {
  const clamped = clamp01(ratio);
  const startRed = (start >> 16) & 0xff;
  const startGreen = (start >> 8) & 0xff;
  const startBlue = start & 0xff;
  const endRed = (end >> 16) & 0xff;
  const endGreen = (end >> 8) & 0xff;
  const endBlue = end & 0xff;
  const red = Math.round(startRed + (endRed - startRed) * clamped);
  const green = Math.round(startGreen + (endGreen - startGreen) * clamped);
  const blue = Math.round(startBlue + (endBlue - startBlue) * clamped);
  return (red << 16) + (green << 8) + blue;
}

export function clamp01(value) {
  return Math.max(0, Math.min(1, value));
}

export function clamp(value, minimum, maximum) {
  return Math.max(minimum, Math.min(maximum, value));
}

export function clampToOrderedRange(value, first, second) {
  return clamp(value, Math.min(first, second), Math.max(first, second));
}

export function formatValue(value, allowFloat = false) {
  if (value == null) return "-";
  if (typeof value === "number" && allowFloat) {
    return Number.isInteger(value) ? String(value) : String(roundValue(value));
  }
  return String(value);
}

export function formatInteger(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return formatValue(value);
  return String(Math.round(numeric)).replace(/\B(?=(\d{3})+(?!\d))/g, ",");
}

export function formatPercent(value) {
  return `${Math.round((value ?? 0) * 100)}%`;
}

export function formatClimateField(value) {
  if (value == null) return "-";
  if (typeof value === "number") return String(roundValue(value));
  return String(value);
}

export function roundValue(value) {
  return Math.round(value * 10000) / 10000;
}

export function hslToRgb(hue, saturation, lightness) {
  if (saturation === 0) {
    const value = Math.round(lightness * 255);
    return [value, value, value];
  }

  const q =
    lightness < 0.5
      ? lightness * (1 + saturation)
      : lightness + saturation - lightness * saturation;
  const p = 2 * lightness - q;
  const convert = (channel) => {
    let t = channel;
    if (t < 0) t += 1;
    if (t > 1) t -= 1;
    if (t < 1 / 6) return p + (q - p) * 6 * t;
    if (t < 1 / 2) return q;
    if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6;
    return p;
  };

  return [
    Math.round(convert(hue + 1 / 3) * 255),
    Math.round(convert(hue) * 255),
    Math.round(convert(hue - 1 / 3) * 255),
  ];
}

export function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}
