#!/usr/bin/env python

import sys

import pandas as pd
import requests

LATITUDE = 33.4672
LONGITUDE = -117.6981

TIDE_STATION_ID = "9410660"

# Replace with company email address (?)
HEADERS = {"User-Agent": "(clean-earth-rovers, ginajung0409@gmail.com)"}
REQUEST_TIMEOUT = 15
LOCAL_TZ = "America/Los_Angeles"


RAIN_ADVISORY_THRESHOLD_MM = 2.54 
RAIN_ADVISORY_WINDOW_DAYS = 3      


LOW_TIDE_THRESHOLD = 0.25


def get_7day_rainfall(lat, lon):
    points_url = f"https://api.weather.gov/points/{lat},{lon}"
    points_res = requests.get(points_url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
    points_res.raise_for_status()
    grid_url = points_res.json()["properties"]["forecastGridData"]

    grid_res = requests.get(grid_url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
    grid_res.raise_for_status()
    precip_layers = grid_res.json()["properties"].get("quantitativePrecipitation", {}).get("values", [])

    if not precip_layers:
        raise ValueError("No quantitativePrecipitation values for this location.")

    records = []
    for item in precip_layers:
        time_block, duration_str = item["validTime"].split("/")
        start_time = pd.to_datetime(time_block, utc=True)
        hours = int(duration_str.replace("PT", "").replace("H", ""))
        value_mm = item["value"] if item["value"] is not None else 0.0
        hourly_value = value_mm / hours if hours else 0.0

        for h in range(hours):
            records.append(
                {"Timestamp": start_time + pd.Timedelta(hours=h), "Rain_Forecast_mm": hourly_value}
            )

    rain_df = pd.DataFrame(records).set_index("Timestamp")


    rain_df.index = rain_df.index.tz_convert(LOCAL_TZ)

    today_local = pd.Timestamp.now(tz=LOCAL_TZ).normalize()
    daily_rain = rain_df.resample("D").sum()
    daily_rain = daily_rain.loc[daily_rain.index >= today_local].head(7)
    daily_rain.index = daily_rain.index.tz_localize(None)
    return daily_rain


def get_7day_tides(station_id):
    now_local = pd.Timestamp.now(tz=LOCAL_TZ)
    start_date = now_local.strftime("%Y%m%d")
    end_date = (now_local + pd.Timedelta(days=7)).strftime("%Y%m%d")
    tide_url = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"

    params = {
        "begin_date": start_date,
        "end_date": end_date,
        "station": station_id,
        "product": "predictions",
        "datum": "mllw",
        "time_zone": "lst_ldt",
        "units": "english",
        "interval": "h",
        "format": "json",
    }

    tide_res = requests.get(tide_url, params=params, timeout=REQUEST_TIMEOUT).json()

    if "predictions" not in tide_res:
        error_msg = tide_res.get("error", {}).get("message", "Unknown API error")
        raise ValueError(
            f"NOAA CO-OPS API failed for station {station_id}: {error_msg} "
            f"(check the station ID is active and supports hourly predictions)"
        )

    tide_df = pd.DataFrame(
        [
            {"Timestamp": pd.to_datetime(item["t"]), "Tide_Height_ft": float(item["v"])}
            for item in tide_res["predictions"]
        ]
    ).set_index("Timestamp")

    daily_tide = tide_df.resample("D").agg(["mean", "min", "max"])["Tide_Height_ft"]
    daily_tide.columns = ["Avg_Tide_ft", "Low_Tide_ft", "High_Tide_ft"]
    return daily_tide.head(7)


def classify_status(forecast, rain_threshold_mm=RAIN_ADVISORY_THRESHOLD_MM,
                                   rain_window_days=RAIN_ADVISORY_WINDOW_DAYS,
                                   low_tide_percentile=LOW_TIDE_THRESHOLD):
    
    rain_trigger = forecast["Expected_Rain_mm"] >= rain_threshold_mm
    advisory_active = rain_trigger.rolling(window=rain_window_days, min_periods=1).max().astype(bool)

    low_tide_cutoff = forecast["Predicted_Low_Tide_ft"].quantile(low_tide_percentile)
    notably_low_tide = forecast["Predicted_Low_Tide_ft"] <= low_tide_cutoff

    risk_score = advisory_active.astype(int) * 2 + notably_low_tide.astype(int)
    status = risk_score.map({0: "SAFE", 1: "MODERATE", 2: "MODERATE", 3: "CRITICAL"})

    detail = pd.Series("", index=forecast.index)
    detail[(~advisory_active) & (~notably_low_tide)] = "No active rain advisory, normal tides"
    detail[(~advisory_active) & notably_low_tide] = "No active rain advisory, notably low tide"
    detail[advisory_active & (~notably_low_tide)] = "Rain advisory active"
    detail[advisory_active & notably_low_tide] = "Rain advisory active, notably low tide"

    return pd.DataFrame({"Status": status, "Status Detail": detail})


def build_forecast(lat=LATITUDE, lon=LONGITUDE, tide_station=TIDE_STATION_ID):
    daily_rain = get_7day_rainfall(lat, lon)
    daily_tide = get_7day_tides(tide_station)

    forecast = pd.DataFrame(index=daily_tide.index)
    forecast["Expected_Rain_mm"] = daily_rain["Rain_Forecast_mm"].reindex(forecast.index, fill_value=0.0)
    forecast["Predicted_Low_Tide_ft"] = daily_tide["Low_Tide_ft"]
    forecast["Predicted_High_Tide_ft"] = daily_tide["High_Tide_ft"]
    status_df = classify_status(forecast)
    forecast = pd.concat([forecast, status_df], axis=1)

    forecast.insert(0, "Forecast Day", forecast.index.strftime("%A (%b %d)"))
    forecast.set_index("Forecast Day", inplace=True)
    return forecast


if __name__ == "__main__":
    try:
        weekly_forecast = build_forecast()
    except requests.exceptions.RequestException as exc:
        sys.exit(f"Network request failed: {exc}")
    except ValueError as exc:
        sys.exit(str(exc))

    pd.set_option("display.width", 120)
    print(weekly_forecast.round(2))