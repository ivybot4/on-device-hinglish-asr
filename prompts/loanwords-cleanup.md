I am building a Hinglish speech recognition dataset. The transcripts come from IIT Bombay Spoken Tutorials: technical tutorials about software, programming (Bash, Python, Linux), LibreOffice, and some science (chemistry, physics). Speakers mix Hindi and English.

In the transcripts, some English words are written in Devanagari instead of Latin script (for example, क्लिक = "click", फ़ाइल = "file"). I want to convert those to Latin script. The attached CSV lists candidate words. A rule-based matcher found them by sound, so many candidates are wrong: they are Hindi words that only sound like an English word (for example, पथ is Hindi for "path", लोग is Hindi for "people", तब is Hindi for "then").

The CSV columns are:
- devanagari: the word as written in the transcripts
- deva_count: how often it occurs in Devanagari
- romanized: a rough rule-based romanization (may be wrong)
- english: the English word the matcher proposed (may be wrong)
- latin_count: how often that English word occurs in Latin script
- similarity: string similarity between romanized and english
- accept: empty

YOUR TASK
For every row, decide yourself. Do not write rule-based or transliteration code to do this; use your own knowledge of Hindi and English.

1. accept = 1 if the Devanagari word IS an English word written in Devanagari (including misspelled or variant spellings, such as क्लिक्क for "click", and plural or -ing forms).
2. accept = 0 if it is a Hindi word, including Hindi words that sound like an English word.
3. Names: Hindi-origin names (रवि, कुमार, अंकित) get 0. English-origin names (जोसेफ = Joseph) get 1.
4. If you cannot decide without the sentence context, give 0 and explain in the note.
5. english_correct: only for rows with accept = 1. Give the correct English spelling, in lowercase. Fix the proposed word when it is wrong (for example, टाइप → "type", not "tip"; लिनक्स → "linux", not "links"; बॉक्स → "box", not "books"). Use the tutorial domain to decide. If one Devanagari word is two English words, give both with a space (for example, "standard in"). Leave it empty for accept = 0.
6. confidence: high, medium, or low.
7. note: a short reason, only when accept = 0 for a word that looks English, when you changed the proposed English word, or when confidence is not high.

OUTPUT
- Return a downloadable CSV file with all the original columns, unchanged, plus three new columns. The column order is: devanagari, deva_count, romanized, english, latin_count, similarity, accept, english_correct, confidence, note
- Fill the existing accept column with 1 or 0. Do not change the values in any other original column.
- Keep every input row, in the same order. Do not skip, merge, or add rows.
- Name the output file as the input file name with "_annotated" added before the extension. For example, loanword_candidates.csv becomes loanword_candidates_annotated.csv.
- Work in batches of about 200 rows until you have done all rows. Do not stop part way and do not summarize instead of labeling.
- At the end, report: the total number of rows, the number of accept = 1 rows, the number of accept = 0 rows, and the number of low-confidence rows.