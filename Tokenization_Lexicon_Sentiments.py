import csv
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.util import ngrams  # NEW ADDITION FOR N-GRAMS
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

nltk.download('punkt', quiet=True)

print("--- Starting Step 1: Lexicon Pipeline with N-Gram Negation Support ---\n")

# 1. Updated Dictionaries with Explicit Bi-grams/Tri-grams
lexicon = {

    "good": {

        "great","excellent","amazing","awesome","wonderful",
        "fantastic","outstanding","perfect","superb",

        "clean","spotless","comfortable","cozy","awesome", "accommodating", "kind", "warm smile",
        "great breakfast",
        "great staff",
        "clean room",
        "updated",
        "quiet",
        "stay again",
        "go to hotel",

        "friendly","helpful","professional",
        "courteous","polite","attentive","welcoming",

        "beautiful","spacious","modern","luxurious",

        "recommend","recommended","enjoyed",
        "love","loved","impressed",

        "exceptional","pleasant","delightful",

        "convenient","quick","efficient",

        "best","value","worth","affordable",

        "delicious","tasty","fresh",

        "peaceful","quiet",

        "great service",
        "excellent service",
        "friendly staff",
        "helpful staff",
        "professional staff",

        "great location",
        "perfect location",

        "clean room",
        "comfortable bed",

        "wonderful stay",
        "fantastic stay",
        "pleasant stay",

        "beautiful view",
        "highly recommend"
    },

    "average": {

        "okay","ok","fine",
        "decent","acceptable",

        "reasonable",
        "satisfactory",

        "normal",
        "standard",
        "basic",

        "nothing special",
        "not bad",
        "not great"
    },

    "medium": {

        "average",
        "fair",
        "ordinary",
        "passable",

        "mediocre",

        "lacking",

        "mixed experience",

        "could be better",
        "average stay",
"normal stay",
"ok stay",
"okay stay",
"it was fine",
"it was okay",
"nothing special",
"nothing exceptional",
"nothing extraordinary",
"nothing to complain",
"no issues",
"no problem",
"not too bad",
"not too good",
"could be improved",
"could improve",
"needs some improvement",
"room for improvement",
"somewhat satisfied",
"partially satisfied",
"mixed feelings",
"mixed review",
"average experience",
"fair experience",
"decent enough",
"acceptable stay",
"a bit expensive",
"a bit old",
"a bit noisy",
"a bit far",
"a little disappointed",
"slightly disappointed",
"not entirely satisfied",
"not fully satisfied",
"expectations were not met fully",
    },

    "worst": {
        "suck",
"sucks",
"smelly",
"bugs",
"bug",
"weired",
"stolen",
"theft",
"robbed",
"broken into",
"would not",
"wouldn't",
"could not",
"couldn't",
"never speak",
"manager unavailable",
"tired staff",
"unhappy",
"unsafe",
"security issue",
"flush",
"toilet issue",
"parking issue",
"towed",

        "bad",
        "terrible",
        "awful",
        "worst",
        "horrible",

        "dirty",
        "filthy",
        "unclean",

        "rude",
        "unprofessional",

        "broken",
        "damaged",

        "outdated",
        "old",
        "worn",

        "stained",

        "smelly",
        "odor",
        "smell",

        "noisy",
        "noise",

        "uncomfortable",

        "slow",
        "delay",
        "delayed",

        "overpriced",
        "expensive",

        "problem",
        "issue",
        "issues",

        "disappointed",
        "disappointing",

        "waste",
        "avoid",

        "unsafe",

        "poor service",
        "bad service",

        "dirty room",
        "dirty bathroom",

        "broken elevator",

        "old furniture",

        "not recommend",
        "never again",

        "waste of money",

        "terrible experience",
        "horrible experience",

        "not good",
        "not clean",
        "not comfortable",
        "not friendly",
        "not nice",
        "not amazing",
        "not perfect","not clean",
"not good",
"not friendly",
"not helpful",
"needs improvement",
"a bit old",
"some issues",
"poor experience",
"not satisfied",
"very disappointed",
"below expectations",
"not worth",
"bad experience",
"not recommend",
"would not stay",
"wouldn't stay",
"never again"
    }
}

analyzer = SentimentIntensityAnalyzer()
input_files = {
    "marriott": "marriott_processed.csv",      
    "hilton": "hilton_processed.csv",          
    "holiday_inn": "holiday_inn_processed.csv"  
}

for brand, file_path in input_files.items():
    print(f"Processing File: {file_path}...")
    file_totals = {"good": 0, "average": 0, "medium": 0, "worst": 0, "neutral": 0, "words": 0}
    rows_to_save = []
    
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f)
            try:
                header = next(reader)
            except StopIteration:
                continue
            
            text_index = 0
            header_lower = [h.lower().strip() for h in header]
            if "review" in header_lower: text_index = header_lower.index("review")
            elif "sentence" in header_lower: text_index = header_lower.index("sentence")
                
            for row in reader:
                if not row or len(row) <= text_index: continue
                sentence_text = row[text_index].replace("\n", " ").replace("\r", " ").strip()

                # Detect non-English / garbled reviews

                english_chars = len(re.findall(r'[a-zA-Z]', sentence_text))
                total_chars = len(sentence_text)

                if total_chars > 0:
                    english_ratio = english_chars / total_chars
                else:
                    english_ratio = 0

                # Less than 70% English → Neutral
                if english_ratio < 0.7:
                    assigned_sentiment = "Neutral"

                    rows_to_save.append([
                        sentence_text,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        assigned_sentiment
                    ])

                    file_totals["neutral"] += 1
                    continue

                # Tokenization
                tokenized_words = word_tokenize(sentence_text.lower())
                sentence_word_count = len(tokenized_words)
                if sentence_word_count == 0: continue
                
                # --- GENERATE BI-GRAMS & TRI-GRAMS ---
                bi_grams = [' '.join(bg) for bg in ngrams(tokenized_words, 2)]
                tri_grams = [' '.join(tg) for tg in ngrams(tokenized_words, 3)]
                
                # Master tokens list containing everything (Uni + Bi + Tri)
                all_tokens = tri_grams + bi_grams + tokenized_words
                
                count_good = 0
                count_average = 0
                count_medium = 0
                count_worst = 0
                
                # Match combinations against updated lexicons
                for token in all_tokens:
                    if token in lexicon["good"]: count_good += 1
                    elif token in lexicon["average"]: count_average += 1
                    elif token in lexicon["medium"]: count_medium += 1.5
                    elif token in lexicon["worst"]: count_worst += 1
                
                counts_dict = {
                    "Good": count_good,
                    "Average": count_average,
                    "Medium": count_medium,
                    "Worst": count_worst
                }

                max_score = max(counts_dict.values())

                if max_score > 0:
                    assigned_sentiment = max(counts_dict, key=counts_dict.get)
                else:
                    vader_score = analyzer.polarity_scores(sentence_text)["compound"]

                    if vader_score >= 0.5:
                        assigned_sentiment = "Good"
                    elif vader_score <= -0.05:
                        assigned_sentiment = "Worst"
                    else:
                        assigned_sentiment = "Average"

                if max_score == 0:
                    vader_score = analyzer.polarity_scores(sentence_text)["compound"]

                    if vader_score >= 0.05:
                        assigned_sentiment = "Good"
                    elif vader_score <= -0.05:
                        assigned_sentiment = "Worst"
                    else:
                        assigned_sentiment = "Average"

                file_totals[assigned_sentiment.lower()] += 1
                file_totals["words"] += sentence_word_count

                vader_score = analyzer.polarity_scores(sentence_text)["compound"]

                rows_to_save.append([
                    sentence_text,
                    sentence_word_count,
                    count_good,
                    count_average,
                    count_medium,
                    count_worst,
                    vader_score,
                    assigned_sentiment
                ])
        output_filename = f"{brand}_calculated_features.csv"
        with open(output_filename, "w", encoding="utf-8", newline="") as out_f:
            writer = csv.writer(out_f)
            writer.writerow([

"review",

"total_word_count",

"count_good",
"count_average",
"count_medium",
"count_worst",

"vader_score",

"sentiments"
])
            writer.writerows(rows_to_save)
            
        print(f"GRAND TOTAL REPORT FOR {brand.upper()} COMPLETELY RE-BUILT WITH N-GRAMS!")
    except FileNotFoundError:
        print(f"ERROR: File not found")

print("Pipeline executed successfully.")