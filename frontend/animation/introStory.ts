export const introStory = [
  { id: "orbit", start: 0, end: .12, title: "EARTH IS CONSTANTLY\nBEING OBSERVED.", detail: "But imagery alone doesn't answer questions." },
  { id: "observation", start: .12, end: .19, title: "WE CAN SEE THE EARTH." },
  { id: "understanding", start: .19, end: .27, title: "UNDERSTANDING IT\nIS HARDER.", detail: "OPTICAL   /   SAR   /   TEMPORAL   /   GEOSPATIAL" },
  { id: "question", start: .27, end: .42, title: "WHAT IF YOU COULD\nASK EARTH A QUESTION?", detail: "“What changed in this region over the last six months?”", query: true },
  { id: "reasoning", start: .42, end: .58, title: "SATQUERY", detail: "Natural language becomes\ngeospatial reasoning.", steps: true },
  { id: "see", start: .58, end: .615, title: "SEE", detail: "Satellite imagery" },
  { id: "compare", start: .615, end: .65, title: "COMPARE", detail: "Changes through time" },
  { id: "reason", start: .65, end: .685, title: "REASON", detail: "Across optical + SAR" },
  { id: "measure", start: .685, end: .72, title: "MEASURE", detail: "With geospatial tools" },
  { id: "from-orbit", start: .72, end: .765, title: "FROM ORBIT…" },
  { id: "places", start: .765, end: .815, title: "…TO THE PLACES\nTHAT MATTER." },
  { id: "evidence", start: .815, end: .86, title: "ONE QUESTION.\nMULTIPLE TOOLS.\nEVIDENCE-BACKED ANSWERS." },
  { id: "final", start: .865, end: .93, title: "ASK EARTH\nA QUESTION." },
] as const;

export const LOGIN_START = .93;
export const clampProgress = (value: number) => Math.max(0, Math.min(1, value));
