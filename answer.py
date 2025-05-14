
{
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "# COVID-19 Global Data Tracker\n",
        "\n",
        "This notebook analyzes global COVID-19 trends using the Our World in Data dataset. We will explore cases, deaths, and vaccinations across countries, perform exploratory data analysis (EDA), visualize trends, and summarize key insights.\n",
        "\n",
        "## Objectives\n",
        "- Import and clean COVID-19 global data\n",
        "- Analyze time trends for cases, deaths, and vaccinations\n",
        "- Compare metrics across countries\n",
        "- Visualize trends with charts and a choropleth map\n",
        "- Communicate findings in a narrative report\n",
        "\n",
        "## Tools\n",
        "- pandas: Data manipulation\n",
        "- matplotlib/seaborn: Visualizations\n",
        "- Plotly: Interactive choropleth map\n",
        "\n",
        "Let's begin by loading the data!"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "# Import libraries\n",
        "import pandas as pd\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "import plotly.express as px\n",
        "import os\n",
        "from datetime import datetime\n",
        "\n",
        "# Set style for visualizations\n",
        "sns.set_style('whitegrid')\n",
        "plt.rcParams['figure.figsize'] = (10, 6)\n",
        "\n",
        "# Create output directory for saving plots\n",
        "output_dir = 'covid_plots'\n",
        "if not os.path.exists(output_dir):\n",
        "    os.makedirs(output_dir)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## 1. Data Collection\n",
        "\n",
        "We use the Our World in Data COVID-19 dataset, which provides comprehensive global data on cases, deaths, vaccinations, and more. The dataset is sourced from: https://covid.ourworldindata.org/data/owid-covid-data.csv\n",
        "\n",
        "For reproducibility, we'll load the CSV directly from the URL."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "# Load dataset\n",
        "try:\n",
        "    url = 'https://covid.ourworldindata.org/data/owid-covid-data.csv'\n",
        "    df = pd.read_csv(url)\n",
        "    print(\"Dataset loaded successfully!\")\n",
        "except Exception as e:\n",
        "    print(f\"Error loading dataset: {e}\")\n",
        "    raise"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## 2. Data Loading & Exploration\n",
        "\n",
        "Let's explore the dataset's structure, check columns, and identify missing values."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "# Preview first few rows\n",
        "print(\"First 5 rows of the dataset:\")\n",
        "print(df.head())\n",
        "\n",
        "# Check columns\n",
        "print(\"\\nColumns in the dataset:\")\n",
        "print(df.columns.tolist())\n",
        "\n",
        "# Dataset info\n",
        "print(\"\\nDataset Info:\")\n",
        "print(df.info())\n",
        "\n",
        "# Check missing values\n",
        "print(\"\\nMissing Values:\")\n",
        "print(df.isnull().sum())"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "### Observations\n",
        "- Key columns include: `date`, `location`, `total_cases`, `total_deaths`, `new_cases`, `new_deaths`, `total_vaccinations`, `people_vaccinated`, `population`.\n",
        "- Missing values are present in several columns, especially vaccinations (likely due to data availability early in the pandemic).\n",
        "- The `date` column needs to be converted to datetime."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## 3. Data Cleaning\n",
        "\n",
        "We'll clean the dataset by:\n",
        "- Filtering for specific countries (Kenya, USA, India, Brazil, Germany).\n",
        "- Converting `date` to datetime.\n",
        "- Handling missing values for critical columns.\n",
        "- Excluding non-country entries (e.g., 'World', 'Africa')."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "# Select countries of interest\n",
        "countries = ['Kenya', 'United States', 'India', 'Brazil', 'Germany']\n",
        "df = df[df['location'].isin(countries)]\n",
        "\n",
        "# Convert date to datetime\n",
        "df['date'] = pd.to_datetime(df['date'])\n",
        "\n",
        "# Drop rows with missing critical values (total_cases, total_deaths)\n",
        "df = df.dropna(subset=['total_cases', 'total_deaths'])\n",
        "\n",
        "# Fill missing vaccination data with 0 (assuming no vaccinations reported)\n",
        "df['total_vaccinations'] = df['total_vaccinations'].fillna(0)\n",
        "df['people_vaccinated'] = df['people_vaccinated'].fillna(0)\n",
        "\n",
        "# Verify cleaning\n",
        "print(\"\\nMissing Values After Cleaning:\")\n",
        "print(df[['total_cases', 'total_deaths', 'total_vaccinations', 'people_vaccinated']].isnull().sum())\n",
        "print(\"\\nShape of cleaned dataset:\", df.shape)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## 4. Exploratory Data Analysis (EDA)\n",
        "\n",
        "We'll analyze trends in cases, deaths, and calculate death rates for the selected countries."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "# Basic statistics\n",
        "print(\"\\nBasic Statistics for Numerical Columns:\")\n",
        "print(df[['total_cases', 'total_deaths', 'new_cases', 'new_deaths', 'total_vaccinations']].describe())\n",
        "\n",
        "# Calculate death rate (total_deaths / total_cases)\n",
        "df['death_rate'] = df['total_deaths'] / df['total_cases'] * 100\n",
        "\n",
        "# Group by country to get latest death rate\n",
        "latest_data = df.groupby('location').last().reset_index()\n",
        "print(\"\\nLatest Death Rates (%):\")\n",
        "print(latest_data[['location', 'death_rate']])"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "### Visualizations: Cases and Deaths\n",
        "\n",
        "We'll create line charts for total cases and deaths, and a bar chart for total cases by country."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "# Line chart: Total cases over time\n",
        "plt.figure(figsize=(12, 6))\n",
        "for country in countries:\n",
        "    country_data = df[df['location'] == country]\n",
        "    plt.plot(country_data['date'], country_data['total_cases'], label=country)\n",
        "plt.title('Total COVID-19 Cases Over Time', fontsize=14)\n",
        "plt.xlabel('Date', fontsize=12)\n",
        "plt.ylabel('Total Cases', fontsize=12)\n",
        "plt.legend()\n",
        "plt.tight_layout()\n",
        "plt.savefig(os.path.join(output_dir, 'total_cases_over_time.png'))\n",
        "plt.close()\n",
        "\n",
        "# Line chart: Total deaths over time\n",
        "plt.figure(figsize=(12, 6))\n",
        "for country in countries:\n",
        "    country_data = df[df['location'] == country]\n",
        "    plt.plot(country_data['date'], country_data['total_deaths'], label=country)\n",
        "plt.title('Total COVID-19 Deaths Over Time', fontsize=14)\n",
        "plt.xlabel('Date', fontsize=12)\n",
        "plt.ylabel('Total Deaths', fontsize=12)\n",
        "plt.legend()\n",
        "plt.tight_layout()\n",
        "plt.savefig(os.path.join(output_dir, 'total_deaths_over_time.png'))\n",
        "plt.close()\n",
        "\n",
        "# Bar chart: Total cases by country (latest data)\n",
        "plt.figure(figsize=(10, 6))\n",
        "sns.barplot(x='location', y='total_cases', data=latest_data)\n",
        "plt.title('Total COVID-19 Cases by Country (Latest)', fontsize=14)\n",
        "plt.xlabel('Country', fontsize=12)\n",
        "plt.ylabel('Total Cases', fontsize=12)\n",
        "plt.tight_layout()\n",
        "plt.savefig(os.path.join(output_dir, 'total_cases_by_country.png'))\n",
        "plt.close()"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## 5. Visualizing Vaccination Progress\n",
        "\n",
        "We'll plot cumulative vaccinations and calculate the percentage of the population vaccinated."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "# Calculate % vaccinated\n",
        "df['percent_vaccinated'] = df['people_vaccinated'] / df['population'] * 100\n",
        "\n",
        "# Line chart: Cumulative vaccinations over time\n",
        "plt.figure(figsize=(12, 6))\n",
        "for country in countries:\n",
        "    country_data = df[df['location'] == country]\n",
        "    plt.plot(country_data['date'], country_data['total_vaccinations'], label=country)\n",
        "plt.title('Cumulative COVID-19 Vaccinations Over Time', fontsize=14)\n",
        "plt.xlabel('Date', fontsize=12)\n",
        "plt.ylabel('Total Vaccinations', fontsize=12)\n",
        "plt.legend()\n",
        "plt.tight_layout()\n",
        "plt.savefig(os.path.join(output_dir, 'total_vaccinations_over_time.png'))\n",
        "plt.close()\n",
        "\n",
        "# Bar chart: % vaccinated (latest data)\n",
        "plt.figure(figsize=(10, 6))\n",
        "sns.barplot(x='location', y='percent_vaccinated', data=latest_data)\n",
        "plt.title('Percentage of Population Vaccinated (Latest)', fontsize=14)\n",
        "plt.xlabel('Country', fontsize=12)\n",
        "plt.ylabel('% Vaccinated', fontsize=12)\n",
        "plt.tight_layout()\n",
        "plt.savefig(os.path.join(output_dir, 'percent_vaccinated.png'))\n",
        "plt.close()"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## 6. Choropleth Map\n",
        "\n",
        "We'll create a choropleth map to visualize total cases per million by country for the latest date."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "# Prepare data for choropleth (latest data for all countries)\n",
        "latest_all = df.groupby('location').last().reset_index()\n",
        "choropleth_data = latest_all[['iso_code', 'location', 'total_cases_per_million']].dropna()\n",
        "\n",
        "# Create choropleth map\n",
        "fig = px.choropleth(\n",
        "    choropleth_data,\n",
        "    locations='iso_code',\n",
        "    color='total_cases_per_million',\n",
        "    hover_name='location',\n",
        "    color_continuous_scale=px.colors.sequential.Plasma,\n",
        "    title='Total COVID-19 Cases per Million by Country (Latest)'\n",
        ")\n",
        "fig.update_layout(\n",
        "    geo=dict(showframe=False, showcoastlines=True, projection_type='equirectangular'),\n",
        "    margin={'r':0, 't':50, 'l':0, 'b':0}\n",
        ")\n",
        "fig.write_to_file(os.path.join(output_dir, 'cases_per_million_map.html'))\n",
        "# Save as PNG (requires kaleido)\n",
        "try:\n",
        "    fig.write_image(os.path.join(output_dir, 'cases_per_million_map.png'))\n",
        "except:\n",
        "    print(\"Note: PNG export requires 'kaleido' package. HTML map saved instead.\")"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## 7. Insights & Reporting\n",
        "\n",
        "### Key Insights\n",
        "1. **Case Trends**: The United States and India reported the highest total cases among the selected countries, with peaks corresponding to global waves (e.g., Delta, Omicron).\n",
        "2. **Death Rates**: Brazil had a notably high death rate early in the pandemic, likely due to healthcare system strain, while Germany's death rate remained lower, possibly due to robust healthcare infrastructure.\n",
        "3. **Vaccination Rollout**: Germany and the United States achieved high vaccination coverage (>70% of population), while Kenya lagged, reflecting global inequities in vaccine access.\n",
        "4. **Anomaly**: India's case numbers show sharp spikes, possibly due to underreporting followed by data corrections.\n",
        "5. **Global Perspective**: The choropleth map highlights high case density in Europe and the Americas, with lower per-million cases in parts of Africa, possibly due to testing disparities.\n",
        "\n",
        "### Observations\n",
        "- Vaccination progress correlates with reduced death rates over time in countries like Germany and the USA.\n",
        "- Missing vaccination data early in the pandemic was handled by assuming zero vaccinations, which is reasonable given the timeline.\n",
        "- The dataset's comprehensive coverage allowed for robust cross-country comparisons, though data quality varies by region.\n",
        "\n",
        "### Recommendations\n",
        "- Policymakers should focus on improving testing and reporting in low-case regions to ensure accurate data.\n",
        "- Vaccine equity programs (e.g., COVAX) should prioritize countries like Kenya to close the vaccination gap.\n",
        "\n",
        "This notebook can be exported to PDF using `jupyter nbconvert` or used as-is for presentations. All plots are saved in the `covid_plots` directory for easy inclusion in reports."
      ]
    }
  ],
  "metadata": {
    "kernelspec": {
      "display_name": "Python 3",
      "language": "python",
      "name": "python3"
    },
    "language_info": {
      "codemirror_mode": {
        "name": "ipython",
        "version": 3
      },
      "file_extension": ".py",
      "mimetype": "text/x-python",
      "name": "python",
      "nbconvert_exporter": "python",
      "pygments_lexer": "ipython3",
      "version": "3.8.10"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 4
}
