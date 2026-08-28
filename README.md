# Park City Trail Conditions

I built this project to answer a pretty simple question:

**Given the recent weather, which trails around Park City are actually worth riding today?**

Weather apps can tell you that it rained yesterday, but that doesn't necessarily tell you what the trails will be like. A lower-elevation, exposed trail might dry quickly while a higher, shaded trail stays wet much longer. After working around mountain biking in Park City, I wanted to see if I could turn those differences into something useful.

The result is a trail conditions dashboard that combines weather, terrain, elevation, and trail data to estimate riding conditions for **373 trails around Park City, Utah**.

The project currently uses a rule-based condition model that I built as a baseline. I'm also collecting real trail reports so that, once I have enough observations, I can test the baseline against actual conditions and start training a supervised model.

## What it does

The dashboard gives each trail:

- a rideability score from 0–100
- an estimated surface condition
- a 7-day rideability forecast
- a trail-specific weather forecast
- distance and elevation information
- an explanation of what's driving the prediction

Trails can be filtered by zone or compared through the rankings page.

There's also a reporting system for recording actual trail conditions. Those observations will eventually become the labeled dataset for the machine-learning side of the project.

## How it works

At a high level, the pipeline looks like this:

```text
Trail geometry + terrain + weather
                |
                v
        Feature engineering
                |
                v
       Condition model v3.2
                |
                v
       Trail-level forecast
                |
                v
        Streamlit dashboard
```

The forecast pipeline runs automatically through GitHub Actions, so new weather data can flow through the model and into the dashboard without me manually updating it every day.

## Building the trail dataset

I started with trail geometry from **OpenStreetMap**.

This ended up being more complicated than just downloading every path tagged as a trail. The raw data contained duplicate segments, disconnected pieces, roads and connectors, hiking paths, and multiple pieces of the same named trail.

I built a processing pipeline to clean those data, remove duplicate directed edges, split disconnected trail components, and review the resulting candidates.

The current catalog contains **373 trails** across Park City and the surrounding trail systems, including:

- Park City / PCMR
- Deer Valley
- Round Valley
- Jeremy Ranch / Glenwild
- Pinebrook
- Summit Park
- Jordanelle
- Wasatch Crest / Millcreek
- Clark Ranch

OSM still isn't perfect — some trails are incomplete or oddly named — but the resulting catalog is much more useful than the raw network.

## Adding terrain

Weather alone isn't enough to describe how a trail dries, so I added elevation and terrain data from the **USGS 3D Elevation Program (3DEP)**.

I use a 10-meter digital elevation model and sample it along each trail to calculate features including:

- minimum, maximum, and mean elevation
- mean and median terrain slope
- terrain aspect distribution

One important distinction: the slope feature represents the slope of the surrounding terrain from the DEM, not necessarily the exact riding grade of the trail.

These features give the model some information about why two trails experiencing similar weather might behave differently.

## Weather

Weather data comes from **Open-Meteo**.

With 373 trails, requesting a completely separate forecast for every trail would be unnecessary and inefficient. I instead grouped nearby trails into **94 localized weather groups**.

Trails within a group share a representative weather location, while each trail keeps its own elevation and terrain features.

The weather pipeline tracks things like:

- daily precipitation
- 3-day and 7-day precipitation
- temperature
- rain and snowfall
- days since precipitation
- recent freeze/thaw behavior
- weather conditions

For forecasts, I pull recent weather along with the next seven days. Including the recent history is important because trail conditions depend on what happened before today, not just today's forecast.

## The dataset

The historical pipeline combines the trail, terrain, and weather data into one trail-by-date dataset.

It currently contains roughly:

**495,000 observations across 373 trails and 1,327 dates**, beginning in January 2023.

That dataset gives me the feature history needed for the current model and, eventually, for training and evaluating learned models.

## Predicting trail conditions

The current model, **v3.2**, is deliberately not a machine-learning model.

It's a domain-informed rule-based baseline that converts recent weather and terrain information into two outputs.

### Surface condition

```text
DUSTY
DRY
IDEAL
DAMP
WET
MUDDY
SNOW
```

### Rideability

```text
EXCELLENT
GOOD
FAIR
POOR
AVOID
```

Each trail also gets a score from 0–100.

I separated surface condition from rideability because they're not quite the same thing. A dry or dusty trail might not have perfect dirt, but it can still be very rideable. A muddy trail is a different story.

The model considers recent and accumulated precipitation, temperature and drying conditions, and terrain characteristics.

The goal of v3.2 isn't to pretend I know the perfect relationship between all of these variables. It's to create a reasonable baseline that I can eventually test against real observations.

## Checking whether the model makes sense

Because I don't yet have enough labeled trail reports to measure real predictive accuracy, I've focused on testing whether the model behaves logically.

The validation scripts check for things like:

- missing predictions
- duplicate trail/date records
- scores outside the 0–100 range
- whether increasing moisture generally reduces rideability
- whether heavy precipitation produces wet/muddy conditions
- whether trails recover as precipitation decreases and drying occurs

I also compare the current forecast against the model's regional predictions.

For example, during a recent validation run, a multi-day rain event caused accumulated precipitation to rise while the regional average rideability score fell substantially. As precipitation tapered off, predicted conditions recovered. New rainfall then pushed the scores down again.

That's the kind of behavior I want to see before worrying about whether the exact score should be a 72 or a 76.

Actual predictive accuracy will require actual trail-condition labels.

## Collecting real conditions

I built a simple reporting system so trail conditions can be recorded by staff, customers, or personal observations.

A report can include:

- trail
- date and time
- condition
- trail section
- source
- notes

Reports are stored in Google Sheets and pulled back into the project as a labeled dataset.

I intentionally don't collect customer names or other personal information because none of that is necessary for the model.

## Why I haven't trained an ML model yet

Machine learning is part of the plan, but I don't think training a model on made-up labels would add anything useful.

Right now, I have a large feature dataset but very few true condition labels.

So the current workflow is:

```text
Build heuristic baseline
        ↓
Collect real trail reports
        ↓
Measure baseline performance
        ↓
Train supervised models
        ↓
Compare them against the baseline
```

Once enough reports accumulate, I want to test models using the weather and terrain features already in the project, use time-aware train/test splits, and see whether a learned model actually beats v3.2.

The condition-report system means the project can gradually move from assumptions about trail behavior toward relationships learned from actual Park City trail conditions.

## Keeping the dashboard current

The production forecast pipeline runs:

```text
Download current weather
        ↓
Build forecast features
        ↓
Run v3.2
        ↓
Generate trail predictions
        ↓
Update dashboard data
```

I use **GitHub Actions** to run the forecast refresh automatically.

One interesting problem here was scale. My original implementation made 94 sequential requests to the weather API. That worked locally but became unreliable on GitHub's hosted runners.

I changed the downloader to batch weather locations together, cutting the number of API requests substantially while keeping all 94 weather regions. That made the automated refresh much more reliable.

The deployed Streamlit dashboard reads the refreshed prediction files, so the public app stays current without requiring a manual local run.

## Project structure

```text
park-city-trail-conditions/
│
├── app/
│   └── trail_conditions_dashboard.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── src/
│   ├── forecast/
│   ├── models/
│   ├── observations/
│   └── weather/
│
├── outputs/
│   └── maps/
│
├── .github/
│   └── workflows/
│
├── requirements.txt
└── README.md
```

Large DEMs, historical datasets, API caches, credentials, and intermediate files are intentionally excluded from the repository.

## Tools

The project is primarily built in Python.

Some of the main tools and libraries I've used are:

**Data:** pandas, NumPy  
**Geospatial:** GeoPandas, OSMnx, Rasterio, Shapely  
**Trail data:** OpenStreetMap  
**Elevation:** USGS 3DEP  
**Weather:** Open-Meteo  
**Dashboard:** Streamlit, Matplotlib  
**Condition reports:** Google Sheets, gspread  
**Automation/deployment:** Git, GitHub Actions, Streamlit Community Cloud

## Limitations

There are still plenty of things I want to improve.

OpenStreetMap isn't a perfect representation of the trail network. Weather stations/API estimates can't capture every hyperlocal difference. The model doesn't directly know soil composition, drainage, tree cover, trail maintenance, or traffic. Some trail geometries also need more cleanup.

Most importantly, v3.2 is still a heuristic model. Until I have enough real condition reports, I can't make a strong claim about its predictive accuracy.

Those limitations are also what make the next stage of the project interesting.

## What's next

The biggest next step is simply **collecting data**.

As real trail reports accumulate, I'll be able to compare predictions with observed conditions, figure out where the current assumptions fail, and start testing supervised models.

Longer term, I'd like the system to learn things like how quickly individual trails dry, which terrain characteristics actually matter most, and how much prediction quality improves when local observations are added to weather and terrain data.

For now, the project is a working end-to-end system:

**trail data → terrain → weather → predictions → automated refresh → interactive dashboard**

with a path toward replacing the hand-built baseline with a model trained on real trail conditions.