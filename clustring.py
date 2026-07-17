import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

print("--- Starting Step 6: Unsupervised Topic-Based K-Means Clustering ---\n")

# 1. Define files to load
files_to_load = {
    "marriott": "marriott_classified.csv",
    "hilton": "hilton_classified.csv",
    "holiday_inn": "holiday_inn_classified.csv"
}

master_reviews = []
master_sentiments = []
master_word_counts = []
master_count_good = []
master_count_average = []
master_count_medium = []
master_count_worst = []
hotel_row_counts = {"marriott": 0, "hilton": 0, "holiday_inn": 0}

# Read inputs safely using native Python CSV streams
for brand, file_path in files_to_load.items():
    print(f"Reading {brand.upper()} sentences...")
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                master_reviews.append(row['review'])
                master_sentiments.append(row['predicted_sentiment'])
                master_word_counts.append(int(row['total_word_count']))
                master_count_good.append(int(row['count_good']))
                master_count_average.append(int(row['count_average']))
                master_count_medium.append(int(row['count_medium']))
                master_count_worst.append(int(row['count_worst']))
                hotel_row_counts[brand] += 1
    except FileNotFoundError:
        print(f"ERROR: Missing file '{file_path}'!")

# -----------------------------------------------------------------
# PHASE 2: TF-IDF VECTORIZATION & UNSUPERVISED K-MEANS
# -----------------------------------------------------------------

print("\nConverting sentences to TF-IDF vectors...")
# We use stop_words='english' to discard meaningless words like 'the', 'is', 'at'
tfidf = TfidfVectorizer(max_features=2000, stop_words='english')
X_tfidf = tfidf.fit_transform(master_reviews)

print("Performing K-Means clustering to discover natural topic groupings...")
# K-Means will group rows purely by shared conversational vocabulary
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_tfidf)
print("K-Means clustering completed successfully!")

# -----------------------------------------------------------------
# OPTIONAL FEATURE: PRINT THE TOP WORDS PER CLUSTER FOR YOUR REPORT
# -----------------------------------------------------------------
print("\nTop words per cluster")
order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
terms = tfidf.get_feature_names_out()
for i in range(5):
    top_words = [terms[ind] for ind in order_centroids[i, :8]]
    print(f"Cluster {i}: Topics related to -> {', '.join(top_words)}")
print("--------------------------------\n")

# -----------------------------------------------------------------
# PHASE 3: WRITE OUT FINAL DATASETS
# -----------------------------------------------------------------

current_index = 0
for brand, count in hotel_row_counts.items():
    output_filename = f"{brand}_unsupervised_topics.csv"
    print(f"  -> Generating: '{output_filename}'")
    
    with open(output_filename, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "review", "total_word_count", "count_good", "count_average", 
            "count_medium", "count_worst", "sentiment", "topic_cluster"
        ])
        
        for i in range(current_index, current_index + count):
            writer.writerow([
                master_reviews[i],
                master_word_counts[i],
                master_count_good[i],
                master_count_average[i],
                master_count_medium[i],
                master_count_worst[i],
                master_sentiments[i],   
                cluster_labels[i]       
            ])
            
    current_index += count

print("\nPipeline execution complete")