import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo 
    import src 
    import matplotlib.pyplot as plt 

    # Use light mode for plot 
    # (Dark mode actually looks good too so comment this out if you want to use dark mode) 
    plt.style.use("default")

    return mo, plt, src


@app.cell(hide_code=True)
def _(mo):
    # Choose observer location 

    observatory_name_input = mo.ui.text(label="Enter observatory name:")
    lat_input = mo.ui.number(label="Latitude (degrees):")
    lon_input = mo.ui.number(label="Longitude (degrees):")

    observatory_presets_radio = mo.ui.radio(
        options={
            "Winer, AZ (32° N, 111° W)": {"observatory_name": "Winer"},
            "Iowa City, IA (42° N, 92° W)": {"lat_long_tuple": (41.66, -91.53)}, 
            "Edmonton, Canada (54° N, 114° W)": {"lat_long_tuple": (53.5, -113.5)}, 
            "Northernmost point in North America (71° N, 157° W)": {"lat_long_tuple": (71.3, -156.8)}, 
        }
    )

    get_observer_button = mo.ui.run_button(label="Submit (set observer location)") 

    observer_tabs = mo.ui.tabs({
        "Observatory Name": 
            mo.md(f"""{observatory_name_input} Ex: Winer"""),
        "Latitude / Longitude": mo.vstack([lat_input, lon_input]),
        "Choose from presets": observatory_presets_radio,
    })

    mo.hstack([observer_tabs, get_observer_button])

    return (
        get_observer_button,
        lat_input,
        lon_input,
        observatory_name_input,
        observatory_presets_radio,
        observer_tabs,
    )


@app.cell(hide_code=True)
def _(mo):
    # Choose target 

    target_name_input = mo.ui.text(label="Enter target name:")
    ra_input = mo.ui.text(label="RA:")
    dec_input = mo.ui.text(label="Dec:")

    target_presets_radio = mo.ui.radio(
        options={
            "Polaris (circumpolar from Winer)": {"target_name": "Polaris"},
            "Epsilon Ursae Minoris (nearly circumpolar from Winer)": {"target_name": "Epsilon Ursae Minoris"}, 
            "Fomalhaut (barely visible from Winer)": {"target_name": "Fomalhaut"}, 
            "Altair (summer target)": {"target_name": "Altair"}, 
            "Betelgeuse (winter target)": {"target_name": "Betelgeuse"}
        }
    )

    get_target_button = mo.ui.run_button(label="Submit (set target)")

    target_tabs = mo.ui.tabs({
        "Target Name": 
            mo.md(f"""{target_name_input} Ex: Altair"""),
        "RA / Dec": 
            mo.md(f"""
                {ra_input} Ex: 17 50 47.4

                {dec_input} Ex: +08 52 06.1 
                """), 
        "Choose from presets": target_presets_radio,
    })

    mo.hstack([target_tabs, get_target_button])
    return (
        dec_input,
        get_target_button,
        ra_input,
        target_name_input,
        target_presets_radio,
        target_tabs,
    )


@app.cell(hide_code=True)
def _(
    get_observer_button,
    lat_input,
    lon_input,
    mo,
    observatory_name_input,
    observatory_presets_radio,
    observer_tabs,
    src,
):
    # Wait until you click the Submit button to recalculate the observer 
    mo.stop(not get_observer_button.value, "Waiting for user to choose observer location")

    if observer_tabs.value == "Observatory Name":
        Observer = src.get_observer(observatory_name=observatory_name_input.value)
    elif observer_tabs.value == "Latitude / Longitude":
        Observer = src.get_observer(lat_long_tuple=(lat_input.value, lon_input.value))
    elif observer_tabs.value == "Choose from presets": 
        Observer = src.get_observer(**observatory_presets_radio.value) 

    Observer 

    return (Observer,)


@app.cell(hide_code=True)
def _(
    dec_input,
    get_target_button,
    mo,
    ra_input,
    src,
    target_name_input,
    target_presets_radio,
    target_tabs,
):
    # Wait until you click the Submit button to recalculate the target  
    mo.stop(not get_target_button.value, "Waiting for user to choose target")

    if target_tabs.value == "Target Name":
        Target = src.get_target(target_name=target_name_input.value)
    elif target_tabs.value == "RA / Dec":
        Target = src.get_target(target_radec_str=ra_input.value + " " + dec_input.value)
    elif target_tabs.value == "Choose from presets": 
        Target = src.get_target(**target_presets_radio.value) 

    Target 
    return (Target,)


@app.cell(hide_code=True)
def _(Observer, Target, plt, src):
    # Calculation 

    try: 
        dates, times, target_alt, sun_alt = src.calc_visibility(Observer, Target) 
        fig = src.plot_visibility(dates, times, target_alt, sun_alt, Target, Observer) 
    except: 
        fig = plt.figure()  

    fig 


    return


if __name__ == "__main__":
    app.run()
