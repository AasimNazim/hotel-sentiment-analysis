from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import csv

app = FastAPI()

# CORS FIX
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

HOTELS = []

with open("master_hotel_data.csv",
          mode="r",
          encoding="utf-8") as file:

    reader = csv.DictReader(file)

    for row in reader:

        HOTELS.append({

            "Hotel_ID": int(row["Hotel_ID"]),
            "Hotel_Brand": row["Hotel_Brand"],
            "Total_Words": int(row["Total_Words"]),

            "Count_Good": int(row["Count_Good"]),
            "Count_Average": int(row["Count_Average"]),
            "Count_Medium": int(row["Count_Medium"]),
            "Count_Worst": int(row["Count_Worst"]),
            "Count_Neutral": int(row["Count_Neutral"]),

            "Cluster_0": int(row["Cluster_0"]),
            "Cluster_1": int(row["Cluster_1"]),
            "Cluster_2": int(row["Cluster_2"]),
            "Cluster_3": int(row["Cluster_3"]),
            "Cluster_4": int(row["Cluster_4"])
        })


@app.get("/hotels")
def get_hotels():

    return HOTELS