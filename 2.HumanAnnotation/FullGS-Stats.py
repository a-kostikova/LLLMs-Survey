import pandas as pd

file_path = "github/3.Human_Annotation/1.AgreementCheck/FullGS/FullGS445.csv"
df = pd.read_csv(file_path, delimiter = ";")

df["Year"] = df["Date"].apply(lambda x: x.split()[-1] if isinstance(x, str) else x)

df["Cleaned_Venue"] = df["Venue"].str.extract(r"([a-zA-Z]+)")

venue_counts = df["Cleaned_Venue"].value_counts().reset_index()
venue_counts.columns = ["Venue", "Count"]

date_counts = df["Year"].value_counts().reset_index()
date_counts.columns = ["Year", "Count"]

final_label_counts = df["FinalLabel"].value_counts().reset_index()
final_label_counts.columns = ["FinalLabel", "Count"]

venue_year_counts = df.groupby(["Cleaned_Venue", "Year"]).size().unstack(fill_value=0).reset_index()

print(venue_counts)
print(date_counts)
print(final_label_counts)
print(venue_year_counts)
