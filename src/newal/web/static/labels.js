"use strict";

// Shared by both modes so the sidebar counter and the training page never
// disagree about what a column means. Plain script, no module, no bundler.

const CAPTURE_LABELS = {
  turns: "전체 턴",
  turns_verified: "검증 통과",
  turns_clean_first_try: "한 번에 통과",
  turns_with_attachments: "첨부 포함",
  repair_pairs: "수리 쌍",
  route_outcomes: "라우팅 결과",
};

function captureLabel(key) {
  return CAPTURE_LABELS[key] || key.replace(/_/g, " ");
}
