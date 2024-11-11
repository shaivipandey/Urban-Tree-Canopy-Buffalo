# Urban-Tree-Canopy-Buffalo
This project analyzes the tree canopy coverage in Buffalo, NY, over the years 2011 to 2021. It explores changes in urban tree canopy, identifying trends, patterns, and neighborhood-specific insights. The findings can potentially be linked to impacts on air quality and urban planning.

## Project Files
1. spatial_analysis.py
Purpose: Analyzes spatial patterns in tree canopy coverage across Buffalo.
Functionality: Identifies areas with high and low tree coverage (hot and cold spots) and checks if canopy patterns are random or clustered.
Output: Generates maps and scatter plots to visualize canopy distribution.

2. neighborhood_analysis.py
Purpose: Examines tree canopy coverage within specific neighborhoods.
Functionality: Summarizes canopy coverage trends for each neighborhood, highlighting significant changes over time.
Output: Provides data summaries for neighborhood-level insights.

3. main.py
Purpose: Main script to run the entire analysis.
Functionality: Integrates results from spatial_analysis.py and neighborhood_analysis.py to produce final outputs.
Output: Generates a comprehensive report and combined visualizations.

## Output
Hot Spot Maps: Visual maps highlighting neighborhoods with high (hot spots) and low (cold spots) tree canopy coverage. These maps show where Buffalo has significant greenery and where it lacks tree coverage.

Yearly Change Maps: Year-over-year visual comparisons that show how tree canopy coverage has changed across the city, indicating areas with noticeable decline or growth.

Moran’s I Scatter Plots: These plots provide a simple way to understand the spatial distribution of tree coverage—whether neighborhoods with high or low canopy are clustered together or randomly distributed.

Trend and Distribution Charts: Graphs showing the overall decline in Buffalo’s tree canopy coverage over time, including year-over-year changes and breakdowns by canopy class.
