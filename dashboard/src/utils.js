// Human-friendly display names for every metric key
export const METRIC_LABELS = {
  faithfulness:        'Fact Accuracy',       // Is every claim in the answer backed by the document?
  answer_relevance:    'Answers the Question', // Does the answer actually address what was asked?
  context_recall:      'Info Coverage',        // Did retrieval find all the relevant information?
  context_precision:   'Source Quality',       // Are the retrieved chunks actually on-topic?
  groundedness:        'Stays in Document',    // Does the answer avoid going beyond the source?
  semantic_similarity: 'Meaning Match',        // How close in meaning is the answer to the expected one?
  rouge_l:             'Word Overlap',         // How many words/phrases match the expected answer?
};

export function metricLabel(key) {
  return METRIC_LABELS[key] ?? key.replace(/_/g, ' ');
}

export function cls(score, threshold = 0.7) {
  if (score >= threshold) return 'pass';
  if (score >= threshold * 0.7) return 'warn';
  return 'fail';
}

export function getDocIcon(name) {
  const ext = name.split('.').pop().toLowerCase();
  if (ext === 'pdf') return '📕';
  if (ext === 'docx' || ext === 'doc') return '📘';
  if (ext === 'txt') return '📄';
  return '📃';
}
