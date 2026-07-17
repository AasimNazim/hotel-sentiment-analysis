from apify_client import ApifyClient

client = ApifyClient("apify_api_g1ukZ5aRcW3qmhNx9f9flWSn8nJZ1z38Zsgy")

run_input = {
    "searchStringsArray": [
        "Holiday Inn hotels USA",
        "Holiday Inn hotels Dubai",
        "Holiday Inn hotels UK",
    ],

    "maxReviews": 500
}

run = client.actor(
    "compass/crawler-google-places"
).call(run_input=run_input)

# OPEN TXT FILE DIRECTLY
with open(
    "holiday_inn_reviews.txt",
    "a",
    encoding="utf-8"
) as f:

    for item in client.dataset(
        run["defaultDatasetId"]
    ).iterate_items():

        if "reviews" in item:

            for r in item["reviews"]:

                text = r.get("text")

                if text:

                    # WRITE IMMEDIATELY
                    f.write(text + "\n\n")

print("DONE")