import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
import numpy as np

# Set font parameters for English display
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.unicode_minus'] = False

def load_data(featured_path, chunk_size=1000000):
    """Load featured data with chunk processing and add season labels"""
    # Initialize accumulators
    season_counts = {"Spring": 0, "Summer": 0, "Fall": 0, "Winter": 0}
    valid_records = 0
    
    # Process in chunks
    print(f"Loading data from {featured_path} in chunks of {chunk_size:,} rows...")
    
    # Define season classification function (Mar-May Spring, Jun-Aug Summer, Sep-Nov Fall, Dec-Feb Winter)
    def get_season(month):
        if 3 <= month <= 5:
            return "Spring"
        elif 6 <= month <= 8:
            return "Summer"
        elif 9 <= month <= 11:
            return "Fall"
        else:
            return "Winter"
    
    # Process each chunk
    chunk_num = 0
    all_chunks = []
    
    for chunk in pd.read_csv(featured_path, low_memory=False, parse_dates=['started_at'], chunksize=chunk_size):
        chunk_num += 1
        print(f"Processing chunk {chunk_num} ({len(chunk):,} rows)...")
        
        # Add month and season columns
        chunk["month"] = chunk["started_at"].dt.month
        chunk["season"] = chunk["month"].apply(get_season)
        
        # Filter invalid routes (start station != end station)
        valid_chunk = chunk[chunk["start_station_name"] != chunk["end_station_name"]].copy()
        
        # Count valid records
        valid_records += len(valid_chunk)
        
        # Count records by season
        season_chunk_counts = valid_chunk["season"].value_counts()
        for season, count in season_chunk_counts.items():
            season_counts[season] += count
        
        # Add route column if not exists
        if "route" not in valid_chunk.columns:
            valid_chunk["route"] = valid_chunk["start_station_name"] + " → " + valid_chunk["end_station_name"]
        
        # Add hour column for rush hour analysis
        valid_chunk["hour"] = valid_chunk["started_at"].dt.hour
        
        all_chunks.append(valid_chunk)
    
    # Combine all chunks
    valid_df = pd.concat(all_chunks, ignore_index=True)
    
    print(f"Data loading completed!")
    print(f"Total valid records: {valid_records:,}")
    print(f"Season distribution:")
    for season, count in season_counts.items():
        print(f"  {season}: {count:,} ({count/valid_records*100:.1f}%)")
    
    return valid_df

def analyze_seasonal_rides(df, save_dir):
    """1. Comparison of ride volume across different seasons"""
    # Count total rides by season
    season_rides = df.groupby("season").size().reindex(["Spring", "Summer", "Fall", "Winter"]).reset_index(name="ride_count")
    
    # Visualization (fix palette warning: explicitly specify hue)
    plt.figure(figsize=(10, 6))
    # Create barplot with wider bars using matplotlib directly for better control
    bars = plt.bar(
        x=season_rides["season"],
        height=season_rides["ride_count"],
        width=0.8,  # Significantly increase bar width
        color=["#4CAF50", "#FF9800", "#2196F3", "#FF5722"]  # Use season colors
    )
    plt.title("Comparison of Total Ride Volume by Season", fontsize=14)
    plt.xlabel("Season", fontsize=12)
    plt.ylabel("Number of Rides", fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    
    # Create custom legend manually
    season_colors = {
        "Spring": "#4CAF50",
        "Summer": "#FF9800",
        "Fall": "#2196F3",
        "Winter": "#FF5722"
    }
    handles = [plt.Rectangle((0,0),1,1, color=color) for season, color in season_colors.items()]
    plt.legend(handles, season_colors.keys(), title="season")
    
    # Add value labels with dynamic offset
    max_value = season_rides["ride_count"].max()
    offset = max_value * 0.02  # 2% of max value for better positioning
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2,  # Center of the bar
            height + offset,  # Above the bar with dynamic offset
            f"{height:,.0f}",  # Formatted value
            ha="center", va="bottom", fontsize=10
        )
    plt.savefig(os.path.join(save_dir, "seasonal_ride_count.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("✅ Seasonal ride count comparison saved")
    return season_rides

def analyze_seasonal_routes(df, save_dir, top_n=5):
    """2. Comparison of popular routes across different seasons (Top N routes)"""
    # Group by season, count rides for each route
    season_routes = {}
    for season in ["Spring", "Summer", "Fall", "Winter"]:
        season_df = df[df["season"] == season]
        top_routes = season_df["route"].value_counts().head(top_n).reset_index()
        top_routes.columns = ["route", "count"]
        season_routes[season] = top_routes
    
    # Visualization (4 subplots for comparison)
    fig, axes = plt.subplots(2, 2, figsize=(24, 20))
    fig.suptitle(f"Top {top_n} Popular Routes by Season", fontsize=18)
    seasons = ["Spring", "Summer", "Fall", "Winter"]
    # Color lists (fixed: pass color list instead of single color code)
    colors = [
        ["#4CAF50"] * top_n,  # Spring: green palette
        ["#FF9800"] * top_n,  # Summer: orange palette
        ["#2196F3"] * top_n,  # Fall: blue palette
        ["#FF5722"] * top_n   # Winter: red palette
    ]
    
    for i, (season, ax) in enumerate(zip(seasons, axes.flat)):
        sns.barplot(
            y="route", 
            x="count", 
            data=season_routes[season], 
            hue="route",  # Explicitly associate y and hue
            palette=colors[i],  # Pass color list
            ax=ax,
            width=0.6  # Increase bar width
        )
        ax.legend().set_visible(False)  # Turn off legend after plotting
        ax.set_title(f"{season} Popular Routes", fontsize=14)
        ax.set_xlabel("Number of Rides", fontsize=12)
        ax.set_ylabel("Route (Start → End)", fontsize=12)
        # Add value labels with dynamic offset based on data scale
        max_value = season_routes[season]["count"].max()
        offset = max_value * 0.01  # 1% of max value for better positioning
        for j, v in enumerate(season_routes[season]["count"]):
            ax.text(v + offset, j, f"{v:,}", va="center", fontsize=10)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Reserve space for title
    plt.savefig(os.path.join(save_dir, f"seasonal_top{top_n}_routes.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Seasonal top {top_n} routes comparison saved")
    return season_routes

def analyze_seasonal_rush_hour(df, save_dir):
    """3. Comparison of morning and evening rush hours across different seasons (7-9am, 5-7pm)"""
    # Mark morning and evening rush hours
    df["hour"] = df["started_at"].dt.hour
    df["rush_hour"] = df["hour"].apply(
        lambda x: "Morning Rush" if 7 <= x <= 9 else ("Evening Rush" if 17 <= x <= 19 else "Off-Peak")
    )
    
    # Count rides by season and rush hour type
    season_rush = df[df["rush_hour"] != "Off-Peak"].groupby(["season", "rush_hour"]).size().unstack()
    season_rush = season_rush.reindex(["Spring", "Summer", "Fall", "Winter"])  # Sort by season
    
    # Visualization (fix palette warning)
    plt.figure(figsize=(12, 7))
    season_rush.plot(
        kind="bar", 
        width=0.7, 
        color=["#FF5722", "#2196F3"],  # Morning rush red, Evening rush blue
        ax=plt.gca()
    )
    plt.title("Comparison of Morning and Evening Rush Hour Rides by Season", fontsize=14)
    plt.xlabel("Season", fontsize=12)
    plt.ylabel("Number of Rides", fontsize=12)
    plt.xticks(rotation=0)
    plt.legend(title="Rush Hour Type")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.savefig(os.path.join(save_dir, "seasonal_rush_hour.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("✅ Seasonal rush hour comparison saved")
    return season_rush

def save_results(season_rides, season_routes, season_rush, save_dir):
    """Save analysis results to CSV"""
    # Seasonal ride counts
    season_rides.to_csv(os.path.join(save_dir, "seasonal_ride_counts.csv"), index=False, encoding="utf-8-sig")
    # Seasonal popular routes
    for season, routes in season_routes.items():
        routes.to_csv(os.path.join(save_dir, f"{season}_popular_routes.csv"), index=False, encoding="utf-8-sig")
    # Seasonal rush hour statistics
    season_rush.to_csv(os.path.join(save_dir, "seasonal_rush_hour_stats.csv"), encoding="utf-8-sig")
    print(f"✅ All analysis results saved to: {save_dir}")

def main():
    # Path configuration (adjust to your project structure)
    featured_path = r"D:/code/502/Bike A/merged_data/cleaned_2023_data.csv"  # Featured data file
    charts_dir = r"D:/code/502/Bike A/result/charts/2023_seasonal_analysis"      # Charts save directory
    results_dir = r"D:/code/502/Bike A/result/charts/2023_seasonal_analysis"      # Results data directory
    
    # Create directories
    os.makedirs(charts_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Execute analysis
    print("===== Seasonal (Quarterly) Ride Data Analysis Started =====")
    df = load_data(featured_path, chunk_size=1000000)
    season_rides = analyze_seasonal_rides(df, charts_dir)
    season_routes = analyze_seasonal_routes(df, charts_dir)
    season_rush = analyze_seasonal_rush_hour(df, charts_dir)
    save_results(season_rides, season_routes, season_rush, results_dir)
    print("===== Seasonal Analysis Completed! =====")

if __name__ == "__main__":
    main()