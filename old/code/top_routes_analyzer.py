import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from haversine import haversine_vector, Unit
import os

# 配置图表样式（确保英文显示清晰）
plt.rcParams['figure.dpi'] = 200  # 提高分辨率
sns.set_style("whitegrid")
sns.set_palette("muted")  # 柔和配色


class TopRouteAnalyzer:
    def __init__(self, data_path, output_dir="./route_results", top_n=15):
        self.data_path = data_path
        self.output_dir = output_dir
        self.top_n = top_n  # 支持自定义Top N数量
        os.makedirs(self.output_dir, exist_ok=True)
        self.df = None
        self.top_routes = None  # 存储Top线路结果
        self.route_details = None  # 存储每条线路的所有距离数据

    def load_data(self, chunk_size=1_000_000):
        """Load data with chunking to handle large files"""
        print("Loading data...")
        dtypes = {
            'start_station_name': 'category',
            'end_station_name': 'category',
            'start_lat': 'float32',
            'start_lng': 'float32',
            'end_lat': 'float32',
            'end_lng': 'float32'
        }
        
        chunks = pd.read_csv(
            self.data_path,
            usecols=list(dtypes.keys()),
            dtype=dtypes,
            chunksize=chunk_size,
            low_memory=False
        )
        
        processed_chunks = []
        for chunk in tqdm(chunks, desc="Processing chunks"):
            # Filter records with complete coordinates
            valid_mask = chunk[['start_lat', 'start_lng', 'end_lat', 'end_lng']].notna().all(axis=1)
            processed_chunks.append(chunk[valid_mask])
        
        self.df = pd.concat(processed_chunks, ignore_index=True)
        print(f"Data loaded successfully. Valid records: {len(self.df):,}")
        return self

    def find_top_routes(self):
        """Identify top N routes by ride frequency"""
        print(f"\nFinding Top {self.top_n} Routes...")
        
        # Generate route identifier (Start → End)
        self.df['route'] = self.df['start_station_name'] + " → " + self.df['end_station_name']
        
        # Count route frequencies and select top N
        route_counts = self.df['route'].value_counts().reset_index()
        route_counts.columns = ['route', 'ride_count']
        self.top_routes = route_counts.head(self.top_n)['route'].tolist()
        print(f"Top {self.top_n} Routes identified.")
        return self

    def calculate_route_distances(self):
        """Calculate distance statistics for top routes"""
        if not self.top_routes:
            raise ValueError("Please run find_top_routes() first")
        
        print("\nCalculating Route Distances...")
        summary_data = []  # For aggregated stats (avg, min, max)
        detail_data = []   # For individual ride distances (for visualization)
        
        for route in tqdm(self.top_routes, desc="Calculating distances"):
            # Extract all records for the current route
            route_records = self.df[self.df['route'] == route][
                ['start_lat', 'start_lng', 'end_lat', 'end_lng', 'route']
            ]
            
            # Batch calculate distances (in kilometers)
            start_coords = route_records[['start_lat', 'start_lng']].values
            end_coords = route_records[['end_lat', 'end_lng']].values
            distances = haversine_vector(start_coords, end_coords, unit=Unit.KILOMETERS)
            
            # Store individual distances
            route_details = pd.DataFrame({
                'route': route,
                'distance_km': distances
            })
            detail_data.append(route_details)
            
            # Calculate aggregated metrics
            summary_data.append({
                'route': route,
                'ride_count': len(route_records),
                'avg_distance_km': round(np.mean(distances), 2),
                'min_distance_km': round(np.min(distances), 2),
                'max_distance_km': round(np.max(distances), 2)
            })
        
        # Organize results
        self.top_routes = pd.DataFrame(summary_data).sort_values('ride_count', ascending=False)
        self.route_details = pd.concat(detail_data, ignore_index=True)
        return self

    def generate_charts(self):
        """Generate practical visualizations: 
        - Frequency vs Distance Bar Chart
        - Heatmap of Ride Count vs Average Distance
        """
        if self.top_routes is None or self.route_details is None:
            raise ValueError("Please run calculate_route_distances() first")
        
        # 1. Frequency and Average Distance Bar Chart
        plt.figure(figsize=(15, 8))
        ax = plt.gca()
        
        # Plot ride count (primary y-axis)
        sns.barplot(
            data=self.top_routes,
            x='route',
            y='ride_count',
            ax=ax,
            alpha=0.8,
            edgecolor='black',
            label='Ride Count'
        )
        ax.set_ylabel('Ride Count', fontsize=14, color='blue')
        ax.tick_params(axis='y', labelcolor='blue')
        
        # Plot average distance (secondary y-axis)
        ax2 = ax.twinx()
        sns.lineplot(
            data=self.top_routes,
            x='route',
            y='avg_distance_km',
            ax=ax2,
            color='red',
            marker='o',
            linewidth=2,
            markersize=8,
            label='Average Distance (km)'
        )
        ax2.set_ylabel('Average Distance (km)', fontsize=14, color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        
        # Chart styling
        ax.set_title(f'Top {self.top_n} Routes: Ride Count vs Average Distance', fontsize=16, pad=20)
        ax.set_xlabel('Route', fontsize=14)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        plt.tight_layout()
        freq_plot_path = os.path.join(self.output_dir, f'top{self.top_n}_routes_freq_distance.png')
        plt.savefig(freq_plot_path, bbox_inches='tight')
        plt.close()
        
        # 2. Scatter Plot: Ride Count vs Average Distance
        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            data=self.top_routes,
            x='ride_count',
            y='avg_distance_km',
            size='ride_count',
            sizes=(50, 200),
            hue='avg_distance_km',
            palette='viridis',
            alpha=0.8,
            edgecolor='black'
        )
        
        # Add annotations for top 3 routes
        for i, row in self.top_routes.head(3).iterrows():
            plt.text(
                row['ride_count'] + 500, 
                row['avg_distance_km'] + 0.1, 
                row['route'], 
                fontsize=8,
                rotation=30,
                ha='left'
            )
        
        # Chart styling
        plt.title('Ride Count vs Average Distance for Top Routes', fontsize=16, pad=20)
        plt.xlabel('Ride Count', fontsize=14)
        plt.ylabel('Average Distance (km)', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        scatter_plot_path = os.path.join(self.output_dir, f'top{self.top_n}_routes_scatter.png')
        plt.savefig(scatter_plot_path, bbox_inches='tight')
        plt.close()
        
        print(f"\nCharts saved to:\n{freq_plot_path}\n{scatter_plot_path}")
        return self

    def save_results(self):
        """Save results to CSV"""
        if self.top_routes is None:
            raise ValueError("Please complete route analysis first")
        
        output_path = os.path.join(self.output_dir, f'top{self.top_n}_routes_with_distance.csv')
        self.top_routes.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")
        return self

    def print_summary(self):
        """Print summary of top routes"""
        print(f"\n===== Top {self.top_n} Routes Summary =====")
        print(self.top_routes[['route', 'ride_count', 'avg_distance_km']].to_string(index=False))
        return self


if __name__ == "__main__":
    # Configure paths (replace with your actual data path)
    DATA_PATH = "../processed_data/2020_featured.csv"  # Path to your data file
    OUTPUT_DIR = "../results/top15_routes_results"               # Output directory (created automatically)
    
    # Run analysis pipeline
    analyzer = TopRouteAnalyzer(
        data_path=DATA_PATH,
        output_dir=OUTPUT_DIR,
        top_n=15
    )
    analyzer.load_data()\
            .find_top_routes()\
            .calculate_route_distances()\
            .save_results()\
            .generate_charts()\
            .print_summary()