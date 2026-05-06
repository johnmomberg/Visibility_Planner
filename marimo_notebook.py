import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo 
    import src 

    return mo, src


@app.cell(hide_code=True)
def _(mo):
    # Choose observer location 

    observatory_name_input = mo.ui.text(label="Enter observatory name")
    lat_input = mo.ui.number(label="Latitude (degrees)")
    lon_input = mo.ui.number(label="Longitude (degrees)")

    get_observer_button = mo.ui.run_button(label="Submit (set observer location)") 

    observer_tabs = mo.ui.tabs({
        "Observatory Name": observatory_name_input,
        "Latitude / Longitude": mo.vstack([lat_input, lon_input]),
        "Choose Preset": mo.md("Preset selection coming soon..."),
    })

    mo.hstack([observer_tabs, get_observer_button])

    return (
        get_observer_button,
        lat_input,
        lon_input,
        observatory_name_input,
        observer_tabs,
    )


@app.cell
def _(mo):
    # Choose target 

    target_name_input = mo.ui.text(label="Enter target name")
    ra_input = mo.ui.text(label="RA:")
    dec_input = mo.ui.text(label="Dec:")

    get_target_button = mo.ui.run_button(label="Submit (set target)")

    target_tabs = mo.ui.tabs({
        "Target Name": target_name_input,
        "RA / Dec": 
            mo.md(f"""
                {ra_input} Ex: 17 50 47.4
            
                {dec_input} Ex: +08 52 06.1 
                """), 
        "Choose Preset": mo.md("Preset selection coming soon..."),
    })

    mo.hstack([target_tabs, get_target_button])
    return (
        dec_input,
        get_target_button,
        ra_input,
        target_name_input,
        target_tabs,
    )


@app.cell(hide_code=True)
def _(
    get_observer_button,
    lat_input,
    lon_input,
    mo,
    observatory_name_input,
    observer_tabs,
    src,
):
    # Wait until you click the Submit button to recalculate the observer 
    mo.stop(not get_observer_button.value, "Waiting for user to choose observer location")

    if observer_tabs.value == "Observatory Name":
        Observer = src.get_observer(observatory_name=observatory_name_input.value)
    elif observer_tabs.value == "Latitude / Longitude":
        Observer = src.get_observer(lat_long_tuple=(lat_input.value, lon_input.value))

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
    target_tabs,
):
    # Wait until you click the Submit button to recalculate the target  
    mo.stop(not get_target_button.value, "Waiting for user to choose target")

    if target_tabs.value == "Target Name":
        Target = src.get_target(target_name=target_name_input.value)
    elif target_tabs.value == "RA / Dec":
        Target = src.get_target(target_radec_str=ra_input.value + " " + dec_input.value)

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
