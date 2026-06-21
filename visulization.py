import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set clean styling for professional project reporting
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# LOAD THE REFINED DATA
df = pd.read_csv('Final_Cleaned_Refined_Unemployment_India.csv')
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')

print("          GENERATING THE THREE CORE PLOTS\n             ")
# Tracking Trends & Spikes Across the Timeline (Rural vs Urban)
plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x='Date', y='Estimated Unemployment Rate (%)', hue='Area', marker='o', errorbar=None, palette='Set1', linewidth=2.5)
plt.axvspan('2020-04-01', '2020-06-30', color='red', alpha=0.12, label='Peak Lockdown Shock Window')
plt.title('Point 1: National Unemployment Trajectory & Pandemic Impact Spikes', fontsize=13, fontweight='bold', pad=15)
plt.xlabel('Timeline', fontsize=11)
plt.ylabel('Average Unemployment Rate (%)', fontsize=11)
plt.legend(title='Sector Type')
plt.tight_layout()
plt.savefig('check_point_1_trends.png', dpi=300)
plt.close()
# Regional Disparities (State-by-State Breakdown)

# Grouping by State to find their absolute baseline averages
state_rank = df.groupby('Region')['Estimated Unemployment Rate (%)'].mean().reset_index().sort_values(by='Estimated Unemployment Rate (%)', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(data=state_rank, x='Estimated Unemployment Rate (%)', y='Region', palette='flare')
plt.title('Point 2: Regional Vulnerability Ranking Across Indian States', fontsize=13, fontweight='bold', pad=15)
plt.xlabel('Average Unemployment Rate (%)', fontsize=11)
plt.ylabel('State / Territory', fontsize=11)
plt.tight_layout()
plt.savefig('check_point_2_regions.png', dpi=300)
plt.close()

# Impact of Labor Participation and Economic Volatility

plt.figure(figsize=(10, 6))
# Scatter plot to evaluate if lower or higher participation tracks with high unemployment rates
sns.scatterplot(data=df, x='Estimated Labour Participation Rate (%)', y='Estimated Unemployment Rate (%)', hue='Is_Lockdown_Period', palette='coolwarm', alpha=0.8, s=70)
plt.title('Point 3: Unemployment Rates vs. Labour Participation Elasticity', fontsize=13, fontweight='bold', pad=15)
plt.xlabel('Estimated Labour Participation Rate (%)', fontsize=11)
plt.ylabel('Estimated Unemployment Rate (%)', fontsize=11)
# Clean up legend descriptions
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(handles, ['Standard Economic Window (0)', 'Lockdown Crisis Phase (1)'], title='Timeline Period')
plt.tight_layout()
plt.savefig('check_point_3_participation.png', dpi=300)
plt.close()

print("\n       ALL THREE DIAGNOSTIC PLOTS COMPLETED SUCCESS     ")
