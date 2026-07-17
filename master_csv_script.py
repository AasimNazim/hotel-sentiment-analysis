import csv
import os


# INPUT FILES
files = [
    "marriott_unsupervised_topics.csv",
    "hilton_unsupervised_topics.csv",
    "holiday_inn_unsupervised_topics.csv"
]


# OUTPUT FILE
output_file = "master_hotel_data.csv"


# FINAL COUNTS STORAGE
hotel_summary = {}


# PROCESS EACH FILE
for file_name in files:

    # HOTEL NAME FROM FILE
    hotel_brand = os.path.splitext(file_name)[0].upper()

    # INITIALIZE COUNTS
    good = 0
    average = 0
    medium = 0
    worst = 0
    neutral = 0

    total_words = 0

    cluster_counts = {
        0: 0,
        1: 0,
        2: 0,
        3: 0,
        4: 0
    }

    # READ CSV
    with open(file_name, "r", encoding="utf-8") as file:

        reader = csv.DictReader(file)

        for row in reader:

            # TOTAL WORDS
            total_words += int(row["total_word_count"])

            # SENTIMENT
            sentiment = row["sentiment"].strip().lower()

            if sentiment == "good":
                good += 1

            elif sentiment == "average":
                average += 1

            elif sentiment == "medium":
                medium += 1

            elif sentiment == "worst":
                worst += 1

            elif sentiment == "neutral":
                neutral += 1

            # CLUSTER
            cluster = int(row["topic_cluster"])

            if cluster in cluster_counts:
                cluster_counts[cluster] += 1

    # SAVE SUMMARY
    hotel_summary[hotel_brand] = {

        "Total_Words": total_words,

        "Count_Good": good,

        "Count_Average": average,

        "Count_Medium": medium,

        "Count_Worst": worst,

        "Count_Neutral": neutral,

        "Cluster_0": cluster_counts[0],
        "Cluster_1": cluster_counts[1],
        "Cluster_2": cluster_counts[2],
        "Cluster_3": cluster_counts[3],
        "Cluster_4": cluster_counts[4]
    }


# WRITE FINAL MASTER CSV
with open(output_file, "w", newline="", encoding="utf-8") as file:

    writer = csv.writer(file)

    # HEADER
    writer.writerow([

        "Hotel_ID",
        "Hotel_Brand",

        "Total_Words",

        "Count_Good",
        "Count_Average",
        "Count_Medium",
        "Count_Worst",
        "Count_Neutral",

        "Cluster_0",
        "Cluster_1",
        "Cluster_2",
        "Cluster_3",
        "Cluster_4"
    ])

    # WRITE DATA
    hotel_id = 1

    for hotel, data in hotel_summary.items():

        writer.writerow([

            hotel_id,
            hotel,

            data["Total_Words"],

            data["Count_Good"],
            data["Count_Average"],
            data["Count_Medium"],
            data["Count_Worst"],
            data["Count_Neutral"],

            data["Cluster_0"],
            data["Cluster_1"],
            data["Cluster_2"],
            data["Cluster_3"],
            data["Cluster_4"]
        ])

        hotel_id += 1


print("Master hotel dataset created successfully!")