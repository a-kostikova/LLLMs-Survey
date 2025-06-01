import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

# Define JSON folders
acl_folder = "New/1.Data_crawling/acl_data"
arxiv_folder = "New/1.Data_crawling/arxiv_data"

# Define ACL venue-to-date mapping
acl_date_mapping = {
    "emnlp2022": "2022-12",
    "iclr2022": "2022-04",
    "tacl2022": "2022-12",
    "naacl2022": "2022-07",
    "iclr2023": "2023-05",
    "acl2022": "2022-05",
    "aacl2022": "2022-09",
    "emnlp2023": "2023-12",
    "tacl2023": "2023-01",
    "acl2023": "2023-07",
    "eacl2023": "2023-05",
    "aacl2023": "2023-11",
    "acl2024": "2024-08",
    "tacl2024": "2024-01",
    "naacl2024": "2024-06",
    "eacl2024": "2024-03",
    "iclr2024": "2024-05",
    "emnlp2024": "2024-11",
}

# Function to extract date from arXiv JSON
def extract_arxiv_date(date_str):
    try:
        return datetime.strptime(date_str[:10], "%Y-%m-%d").strftime("%Y-%m")
    except Exception:
        return None

# Collect paper counts separately for ACL and ArXiv
acl_date_counts = {}
arxiv_date_counts = {}

# Process ACL data
for file in os.listdir(acl_folder):
    if file.endswith(".json"):
        venue_name = file.replace(".json", "")
        date = acl_date_mapping.get(venue_name, None)
        if date:
            with open(os.path.join(acl_folder, file), "r", encoding="utf-8") as f:
                papers = json.load(f)
                num_papers = len(papers)
                if num_papers > 0:  # Only keep non-zero values
                    acl_date_counts[date] = acl_date_counts.get(date, 0) + num_papers

# Process ArXiv data
for file in os.listdir(arxiv_folder):
    if file.endswith(".json"):
        with open(os.path.join(arxiv_folder, file), "r", encoding="utf-8") as f:
            papers = json.load(f)
            for paper in papers:
                date = extract_arxiv_date(paper.get("published", ""))
                if date:
                    arxiv_date_counts[date] = arxiv_date_counts.get(date, 0) + 1  # Increment count

# Convert to DataFrames for sorting and plotting
df_acl = pd.DataFrame(list(acl_date_counts.items()), columns=["Date", "ACL_Count"])
df_acl["Date"] = pd.to_datetime(df_acl["Date"])
df_acl = df_acl.sort_values("Date")

df_arxiv = pd.DataFrame(list(arxiv_date_counts.items()), columns=["Date", "ArXiv_Count"])
df_arxiv["Date"] = pd.to_datetime(df_arxiv["Date"])
df_arxiv = df_arxiv.sort_values("Date")

# Merge both datasets on Date
df = pd.merge(df_acl, df_arxiv, on="Date", how="outer").fillna(0)
df = df.sort_values("Date")

# Generate a complete range of months from first available month to December 2024
end_date = pd.Timestamp("2025-03-28")  # Set end of x-axis to 28 March 2025
all_dates = pd.date_range(start=df["Date"].min(), end=end_date, freq="MS")  # 'MS' = Month Start
df = df.set_index("Date").reindex(all_dates, fill_value=0).reset_index().rename(columns={"index": "Date"})

# Compute total count (ACL + ArXiv)
df["Total_Count"] = df["ACL_Count"] + df["ArXiv_Count"]

# Define cutoff dates for ArXiv and Total
arxiv_cutoff_date = pd.Timestamp("2025-03-28")  # ArXiv stops in October 2024
total_cutoff_date = pd.Timestamp("2025-03-28")  # Total line stops in November 2024

# Mask ArXiv and Total data beyond the cutoffs
df["ArXiv_Plot"] = df["ArXiv_Count"].where(df["Date"] <= arxiv_cutoff_date, other=None)
df["Total_Plot"] = df["Total_Count"].where(df["Date"] <= total_cutoff_date, other=None)

# Determine y-axis maximum
y_max = df["Total_Count"].max()

# Plot the chart
plt.figure(figsize=(6, 3.5))  # Reduce figure size
bar_width = 15  # Reduce bar width

# ACL as bars
plt.bar(df["Date"], df["ACL_Count"], label="ACL Papers", color="blue", alpha=0.7, width=bar_width)

# Add conference labels on top of ACL bars
for date, count in zip(df["Date"], df["ACL_Count"]):
    if count > 0:  # Only label bars that have papers
        conf_name = [k for k, v in acl_date_mapping.items() if v == date.strftime("%Y-%m")]
        if conf_name:
            short_name = conf_name[0][:-4] + "'" + conf_name[0][-2:]  # Abbreviate years
            plt.text(date, count + (y_max * 0.03), short_name, fontsize=7, ha="center", rotation=90, color="black")

# ArXiv as a red line (cut off after October 2024)
plt.plot(df["Date"], df["ArXiv_Plot"], label="ArXiv Papers", color="red", marker="o", linestyle="-", linewidth=1.5, alpha=0.8)

# Total (ACL + ArXiv) as a dimmed line (cut off after November 2024)
plt.plot(df["Date"], df["Total_Plot"], label="Total Papers", color="gray", linestyle="--", linewidth=1.5, alpha=0.7)

# Formatting x-axis for better readability
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))  # Format as Month Year
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))  # Show labels every 2 months
plt.xlim(df["Date"].min(), end_date)  # Limit x-axis to range from first data to December 2024

# Adjust y-axis height
plt.ylim(0, y_max * 1.05)  # Reduce padding

# Labels and formatting
#plt.xlabel("Date", fontsize=10)
plt.ylabel("Number of Papers", fontsize=9)
plt.title("ACL vs ArXiv Paper Distribution", fontsize=10.5)
plt.grid(axis="both", linestyle="--", alpha=0.3)  # Enables both horizontal and vertical grids

plt.xticks(rotation=45, fontsize=7)
plt.yticks(fontsize=7)

# Reduce legend size
plt.legend(fontsize=8, loc="upper left")

# Show the plot
plt.tight_layout()  # Optimize spacing
plt.savefig("New/1.Data_crawling/ACLvsArxivPaperDistribution_NEW.pdf", format="pdf", bbox_inches="tight")

# ---- Print paper totals ----
total_acl = int(df["ACL_Count"].sum())
total_arxiv = int(df["ArXiv_Count"].sum())
total_all = total_acl + total_arxiv

print(f"✅ Total ACL papers: {total_acl}")
print(f"✅ Total ArXiv papers: {total_arxiv}")
print(f"📊 Total papers combined: {total_all}")

from collections import defaultdict

# --- Count ACL papers by year ---
acl_year_counts = defaultdict(int)
for file in os.listdir(acl_folder):
    if file.endswith(".json"):
        venue = file.replace(".json", "")
        if venue in acl_date_mapping:
            year = acl_date_mapping[venue].split("-")[0]
            with open(os.path.join(acl_folder, file), "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                    acl_year_counts[year] += len(data)
                except json.JSONDecodeError:
                    print(f"Error reading {file}, skipping...")

# --- Count ArXiv papers by year ---
arxiv_year_counts = defaultdict(int)
for file in os.listdir(arxiv_folder):
    if file.endswith(".json"):
        with open(os.path.join(arxiv_folder, file), "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                for paper in data:
                    date = paper.get("published", "")
                    if len(date) >= 4:
                        year = date[:4]
                        arxiv_year_counts[year] += 1
            except json.JSONDecodeError:
                print(f"Error reading {file}, skipping...")

# --- Print summary ---
print("\n📚 ACL Papers by Year:")
for year in sorted(acl_year_counts):
    print(f"{year}: {acl_year_counts[year]} papers")

print("\n📚 ArXiv Papers by Year:")
for year in sorted(arxiv_year_counts):
    print(f"{year}: {arxiv_year_counts[year]} papers")

plt.show()
