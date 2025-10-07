# README — Z-algorithm Pattern Matcher

### Overview
This program finds all exact occurrences of a pattern p in a text t using the Z-algorithm and reports:
- the 1-based starting positions of every match (one per line, ascending), then
- three tallies on separate lines:
    - Number of comparisons: <int>
    - Number of matches: <int>
    - Number of mismatches: <int>
A “comparison” is a single character == check performed while extending Z-boxes. Each comparison is counted as either a match or a mismatch.

### Files
exact_pattern_matching.py — main program (Z-algorithm implementation + CLI)

### Requirements
- Python 3.8+ (no external packages)
- POSIX or Windows shell

### Input format
Each input file must contain exactly two lines:
- Line 1: the text t
- Line 2: the pattern p

Example:
ATATTGATGATATG...
ATTGATGATA

### Output format
- One line per match position (1-based indices into t)
- Then the three tallies, each on its own line

Example:
3
166
538
688
1111
Number of comparisons: 1879
Number of matches: 522
Number of mismatches: 1357

### How to run
Print to stdout
python exact_pattern_matching.py <input_file>

Example: python zmatch.py samples/sample_0

Redirect to a file if needed:
python exact_pattern_matching.py samples/sample_0 > sol_0

Write directly to an output file
python exact_pattern_matching.py <input_file> <output_file>

Example:
python exact_pattern_matching.py samples/sample_0 sol_0

The program does not create any directories, therefore the path to <output_file> must exist.

### Batch over many samples
From a shell inside the directory that contains samples/:

for f in samples/sample_*; do
  base=$(basename "$f")
  out="sol_${base#sample_}"
  python zmatch.py "$f" "$out"
done

### Notes on correctness
- Positions are 1-based: first character of t is position 1; last is position |t|.
- The algorithm uses the standard pattern + '$' + text construction (the sentinel $ is not in the DNA alphabet), so there are no separator collisions.
- We skip unnecessary extensions when a Z value is fully determined inside the current [L, R] window; this keeps comparison counts tight.

### Implementation details
- Z-array construction maintains [L, R]. When i ≤ R:
    - If Z[i-L] < R-i+1, we reuse Z[i] = Z[i-L] and do not extend (no new comparisons).
    - Otherwise, we set Z[i] = R-i+1 and extend from R+1.
- Every time we compare t[a] with t[i+a], we increment the global comparison counter and classify it as a match or mismatch.