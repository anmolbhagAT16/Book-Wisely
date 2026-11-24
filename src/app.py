from pathlib import Path
from datetime import date

import joblib
import numpy as np
import pandas as pd
import streamlit as st


 
# 1. PROJECT PATHS

PROJECT_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_DIR
    / "Models"
    / "bookwisely_all_models.joblib"
)

ROUTE_PATH = (
    PROJECT_DIR
    / "Data"
    / "processed"
    / "route_stops.parquet"
)


 
# 2. CATEGORY DISPLAY NAMES


CATEGORY_NAMES = {
    "AMTB": "Amrit Bharat Express",
    "DMU": "Diesel Multiple Unit",
    "DRNT": "Duronto Express",
    "EMU": "Electric Multiple Unit",
    "GBR": "Garib Rath",
    "INGL": "Description not verified",
    "JSH": "Jan Shatabdi",
    "MEMU": "Mainline Electric Multiple Unit",
    "MEX": "Mail/Express",
    "PAS": "Passenger",
    "PEXP": "Parcel Express (Non-Passenger)",
    "RAJ": "Rajdhani Express",
    "SHT": "Shatabdi Express",
    "SUB": "Suburban",
    "SUF": "Superfast",
    "TEJ": "Tejas Express",
    "TOD": "Train on Demand",
    "VNDB": "Vande Bharat Express",
    "VNDM": "Namo Bharat Rapid Rail",
    "VNDS": "Vande Bharat Sleeper"
}


def category_label(code):

    description = CATEGORY_NAMES.get(
        code,
        "Description unavailable"
    )

    return f"{code} — {description}"


 
# 3. STREAMLIT CONFIGURATION

st.set_page_config(
    page_title="BookWisely - Train Delay Predictor",
    layout="wide"
)

st.title("Book-Wisely")

st.subheader("Train Arrival Delay Predictor")

st.write(
    "Select a journey date, source station, and destination "
    "station to discover matching train services and predict "
    "their arrival delays."
)

st.divider()


 
# 4. LOAD SAVED MODELS AND ROUTE DATA

@st.cache_resource
def load_model_bundle():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


@st.cache_data
def load_route_data():

    if not ROUTE_PATH.exists():
        raise FileNotFoundError(
            f"Route file not found: {ROUTE_PATH}"
        )

    return pd.read_parquet(ROUTE_PATH)


try:

    bundle = load_model_bundle()
    route_data = load_route_data()

except Exception as error:
    st.error(
        f"Application data could not be loaded: {error}"
    )
    st.stop()


# Retrieve saved models and preprocessing
models = bundle["models"]

preprocessor = bundle["preprocessor"]

feature_columns = bundle["feature_columns"]


 
# 5. PREPARE AVAILABLE STATIONS AND CATEGORIES
 

categorical_encoder = (
    preprocessor.named_transformers_["categorical"]
)

# Categories recognized by the trained model
trained_categories = {
    str(value).upper()
    for value in categorical_encoder.categories_[0]
}

# Destination stations recognized by the trained model
trained_stations = {
    str(value).upper()
    for value in categorical_encoder.categories_[1]
}


# Standardize route data
route_data = route_data.copy()

route_data["station_code"] = (
    route_data["station_code"]
    .astype("string")
    .str.strip()
    .str.upper()
)

route_data["category"] = (
    route_data["category"]
    .astype("string")
    .str.strip()
    .str.upper()
)

route_data["train_number"] = (
    route_data["train_number"]
    .astype("string")
    .str.strip()
)


# Convert start_date into a date format
route_data["start_date"] = pd.to_datetime(
    route_data["start_date"],
    errors="coerce"
).dt.date


# Exclude parcel services from passenger route search
passenger_routes = route_data.loc[
    route_data["category"] != "PEXP"
].copy()


# Source options
source_options = sorted(
    passenger_routes["station_code"]
    .dropna()
    .unique()
    .tolist()
)


# Destination must also be recognized by the trained model
destination_options = sorted(
    set(source_options).intersection(
        trained_stations
    )
)


# Category options available in the trained model
category_options = sorted(
    trained_categories
)


 
# 6. APPLICATION INPUTS
 

st.header("Find Your Train Route")


with st.form("route_search_form"):

    
    # FUTURE JOURNEY DATE
    

    st.subheader("Select Journey Date")

    selected_journey_date = st.date_input(
        "Train journey date",
        value=date.today(),
        min_value=date.today(),
        help=(
            "Choose today or a future date. Operating weekdays "
            "are estimated from historical route records."
        )
    )

    selected_weekday = (
        selected_journey_date.strftime("%A")
    )

    st.caption(
        "The selected date is used to identify trains that "
        "were historically observed operating on that weekday. "
        "This is not a live timetable."
    )

    st.divider()


    
    # SOURCE AND DESTINATION STATIONS
    

    col1, col2 = st.columns(2)

    with col1:

        source_station = st.selectbox(
            "Source Station",
            options=source_options,
            index=None,
            placeholder="Type source station code...",
            filter_mode="contains"
        )

    with col2:

        destination_station = st.selectbox(
            "Destination Station",
            options=destination_options,
            index=None,
            placeholder="Type destination station code...",
            filter_mode="contains"
        )

    st.caption(
        "Both station fields support searching with uppercase "
        "or lowercase letters."
    )


    
    # TRAIN CATEGORY
    

    category_choice = st.selectbox(
        "Train Category",
        options=["ALL"] + category_options,
        index=0,
        format_func=lambda code: (
            "All passenger categories"
            if code == "ALL"
            else category_label(code)
        ),
        filter_mode=None
    )

    st.caption(
        "PEXP (Parcel Express) is not included in passenger "
        "route search."
    )

    st.divider()


    
    # MODEL SELECTION
    

    model_choice = st.radio(
        "Select Prediction Model",
        options=[
            "Compare All Three Models",
            "Linear Regression",
            "Random Forest",
            "HistGradientBoosting"
        ],
        horizontal=True
    )


    
    # SEARCH BUTTON
    

    search_submitted = st.form_submit_button(
        "Find Trains and Predict Arrival Delay",
        use_container_width=True
    )


 
# 7. FIND MATCHING TRAIN ROUTES
 

if search_submitted:

    if (
        source_station is None
        or destination_station is None
    ):

        st.error(
            "Please select both source and destination stations."
        )

        st.stop()


    if source_station == destination_station:

        st.error(
            "Source and destination stations must be different."
        )

        st.stop()


    if category_choice == "PEXP":

        st.error(
            "Parcel Express is a non-passenger service. "
            "Please select a passenger train category."
        )

        st.stop()


    
    # MATCHING KEYS
    

    run_columns = [
        "train_number",
        "train_name",
        "category",
        "start_date"
    ]


    
    # SOURCE STATION RECORDS
    

    source_stops = passenger_routes.loc[
        passenger_routes["station_code"]
        == source_station,
        run_columns + ["seq"]
    ].copy()

    source_stops = source_stops.rename(
        columns={
            "seq": "source_sequence"
        }
    )


    
    # DESTINATION STATION RECORDS
    

    destination_stops = passenger_routes.loc[
        (
            passenger_routes["station_code"]
            == destination_station
        )
        &
        (
            passenger_routes[
                "scheduled_arrival_hour"
            ].notna()
        ),
        run_columns
        + [
            "seq",
            "scheduled_arrival_hour",
            "scheduled_arrival_time"
        ]
    ].copy()

    destination_stops = destination_stops.rename(
        columns={
            "seq": "destination_sequence"
        }
    )


    
    # MATCH SOURCE AND DESTINATION IN SAME TRAIN RUN
    

    matched_routes = source_stops.merge(
        destination_stops,
        on=run_columns,
        how="inner"
    )


    # Destination must occur after source
    matched_routes = matched_routes.loc[
        matched_routes["destination_sequence"]
        > matched_routes["source_sequence"]
    ].copy()


    
    # APPLY CATEGORY FILTER
    

    if category_choice != "ALL":

        matched_routes = matched_routes.loc[
            matched_routes["category"]
            == category_choice
        ].copy()


    if matched_routes.empty:

        st.warning(
            f"No matching train services were found from "
            f"{source_station} to {destination_station} "
            f"in the available historical sample."
        )

        st.stop()


    
    # 8. ESTIMATE TRAIN OPERATING WEEKDAYS
    

    service_keys = [
        "train_number",
        "train_name",
        "category"
    ]


    # Convert historical start dates into weekday names
    matched_routes["operating_weekday"] = (
        pd.to_datetime(
            matched_routes["start_date"],
            errors="coerce"
        ).dt.day_name()
    )


    # Summarize historical operating patterns
    historical_route_stats = (
        matched_routes
        .groupby(
            service_keys,
            as_index=False
        )
        .agg(
            historical_runs=(
                "start_date",
                "nunique"
            ),

            first_sample_date=(
                "start_date",
                "min"
            ),

            last_sample_date=(
                "start_date",
                "max"
            ),

            observed_weekdays=(
                "operating_weekday",
                lambda values: ", ".join(
                    day
                    for day in [
                        "Monday",
                        "Tuesday",
                        "Wednesday",
                        "Thursday",
                        "Friday",
                        "Saturday",
                        "Sunday"
                    ]
                    if day in set(
                        values.dropna()
                    )
                )
            )
        )
    )


    
    # FILTER FOR SELECTED JOURNEY WEEKDAY
    

    matched_routes = matched_routes.loc[
        matched_routes["operating_weekday"]
        == selected_weekday
    ].copy()


    if matched_routes.empty:

        st.warning(
            f"No matching services from {source_station} "
            f"to {destination_station} were observed operating "
            f"on {selected_weekday} in this historical sample. "
            "Try another date or weekday. This does not prove "
            "that no train operates on that day."
        )

        st.stop()


    
    # 9. SUMMARIZE MATCHING TRAIN SERVICES
    

    def most_common_value(series):

        values = series.dropna().mode()

        if len(values) == 0:
            return np.nan

        return values.iloc[0]


    route_summary = (
        matched_routes
        .groupby(
            [
                "train_number",
                "train_name",
                "category"
            ],
            as_index=False
        )
        .agg(
            destination_sequence=(
                "destination_sequence",
                most_common_value
            ),

            scheduled_arrival_hour=(
                "scheduled_arrival_hour",
                most_common_value
            ),

            scheduled_arrival_time=(
                "scheduled_arrival_time",
                most_common_value
            )
        )
    )


    # Add full historical operating pattern information
    route_summary = route_summary.merge(
        historical_route_stats,
        on=service_keys,
        how="left"
    )


    route_summary = route_summary.dropna(
        subset=[
            "destination_sequence",
            "scheduled_arrival_hour"
        ]
    ).copy()


    if route_summary.empty:

        st.warning(
            "Matching routes were found, but their destination "
            "arrival times were unavailable."
        )

        st.stop()


    route_summary["destination_sequence"] = (
        route_summary[
            "destination_sequence"
        ].astype(int)
    )


    route_summary["scheduled_arrival_hour"] = (
        route_summary[
            "scheduled_arrival_hour"
        ].astype(int)
    )


    
    # 10. PREPARE MODEL INPUT FEATURES
    

    model_inputs = pd.DataFrame({

        "category": (
            route_summary["category"].astype(str)
        ),

        "seq": (
            route_summary["destination_sequence"]
        ),

        "station_code": (
            destination_station
        ),

        "scheduled_arrival_hour": (
            route_summary[
                "scheduled_arrival_hour"
            ]
        )
    })


    model_inputs = model_inputs[
        feature_columns
    ]


    
    # 11. GENERATE PREDICTIONS
    

    predictions = {}


    # HistGradientBoosting uses its own Pipeline
    if model_choice in [
        "Compare All Three Models",
        "HistGradientBoosting"
    ]:

        hist_predictions = models[
            "HistGradientBoosting"
        ].predict(
            model_inputs
        )

        predictions[
            "HistGradientBoosting"
        ] = hist_predictions


    # Linear Regression and Random Forest use
    # the saved One-Hot Encoder
    if model_choice in [
        "Compare All Three Models",
        "Linear Regression",
        "Random Forest"
    ]:

        encoded_inputs = (
            preprocessor.transform(
                model_inputs
            )
        )


        selected_models = [
            "Linear Regression",
            "Random Forest"
        ]


        if model_choice != "Compare All Three Models":

            selected_models = [
                model_choice
            ]


        for name in selected_models:

            predictions[name] = (
                models[name].predict(
                    encoded_inputs
                )
            )


    
    # 12. DISPLAY TRAIN SERVICES AND PREDICTIONS
    

    st.divider()

    st.header("Available Train Services")


    st.write(
        f"Route: **{source_station} → "
        f"{destination_station}**"
    )


    st.write(
        f"Journey date: **"
        f"{selected_journey_date.strftime('%d %B %Y')}**"
    )


    st.write(
        f"Selected weekday: **{selected_weekday}**"
    )


    st.success(
        f"{len(route_summary)} matching train service(s) found."
    )


    st.caption(
        "Services are filtered by the selected weekday using "
        "historical run dates. Operating days are inferred "
        "from the sample and are not verified future schedules."
    )


    
    # BUILD RESULTS TABLE

    results = pd.DataFrame({

        "Train Number": (
            route_summary["train_number"]
        ),

        "Train Name": (
            route_summary["train_name"]
        ),

        "Category": (
            route_summary["category"].apply(
                category_label
            )
        ),

        # "Destination Sequence": (
        #     route_summary[
        #         "destination_sequence"
        #     ]
        # ),

        "Scheduled Arrival (IST)": (
            route_summary[
                "scheduled_arrival_time"
            ]
        ),

        # "Inferred Operating Days": (
        #     route_summary[
        #         "observed_weekdays"
        #     ]
        # ),

        # "Historical Runs (Sample)": (
        #     route_summary[
        #         "historical_runs"
        #     ]
        # )
    })


    # Add model predictions to results
    for name, values in predictions.items():

        results[
            f"{name} Prediction (min)"
        ] = np.round(
            values,
            2
        )


    # SORT RESULTS

    if model_choice == "Compare All Three Models":

        results = results.sort_values(
            "Linear Regression Prediction (min)"
        )

    else:

        prediction_column = (
            f"{model_choice} Prediction (min)"
        )

        results = results.sort_values(
            prediction_column
        )


    results = results.reset_index(
        drop=True
    )


    
    # DISPLAY TRAIN TABLE
    

    st.dataframe(
        results,
        use_container_width=True,
        hide_index=True
    )


    
    # 13. BEST OPTION — MINIMUM PREDICTED DELAY

    if model_choice == "Compare All Three Models":

        best_prediction_column = (
            "Linear Regression Prediction (min)"
        )

        best_model_label = (
            "Linear Regression"
        )

    else:

        best_prediction_column = (
            f"{model_choice} Prediction (min)"
        )

        best_model_label = (
            model_choice
        )


    # Find train with minimum predicted delay
    best_option = results.loc[
        results[
            best_prediction_column
        ].idxmin()
    ]


    st.subheader(
        "Best Option — Minimum Predicted Delay"
    )


    best_col1, best_col2, best_col3 = (
        st.columns([2, 2, 1])
    )


    best_col1.metric(
        "Recommended Train",
        str(
            best_option["Train Number"]
        )
    )


    best_col2.metric(
        "Predicted Arrival Delay",
        f"{best_option[best_prediction_column]:.2f} min"
    )


    best_col3.metric(
        "Model",
        best_model_label
    )


    st.success(
        f"**{best_option['Train Name']}** "
        f"({best_option['Category']}) is the best option "
        f"according to {best_model_label}, with the lowest "
        f"predicted arrival delay among the displayed services."
    )


    st.caption(
        "This recommendation is based on model predictions "
        "for the selected route and date, not live "
        "train-running information."
    )


    
    # 14. ROUTE SEARCH SUMMARY

    st.divider()

    st.subheader(
        "Route Search Summary"
    )


    col1, col2, col3 = st.columns(3)


    col1.metric(
        "Matching Train Services",
        len(route_summary)
    )


    col2.metric(
        "Source Station",
        source_station
    )


    col3.metric(
        "Destination Station",
        destination_station
    )


    st.caption(
        "Inferred Operating Days shows weekdays on which "
        "this service was observed on the selected route "
        "in the historical sample. Historical Runs (Sample) "
        "counts distinct recorded start dates. Neither value "
        "confirms a future timetable."
    )


    st.info(
        "Predictions estimate arrival delay at the selected "
        "destination station using train category, destination "
        "sequence, station code, and scheduled arrival hour. "
        "They are not live delay predictions."
    )


 
# 15. FOOTER
 
st.divider()

st.caption(
    "Book-Wisely | Machine Learning Based Train Arrival "
    "Delay Prediction | Educational Project"
)
