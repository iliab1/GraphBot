import requests
import pandas as pd
import streamlit as st
from st_aggrid import AgGrid, GridUpdateMode, DataReturnMode
from st_aggrid.grid_options_builder import GridOptionsBuilder
from auth import login, login_sidebar_button
from urllib.parse import urlencode

login_sidebar_button()


if "cookies" not in st.session_state:
    login()
    st.stop()


# Function to fetch data from the API
def get_sources():
    if "cookies" in st.session_state:
        try:
            url = "http://backend:8080/sources_list"
            response = requests.post(url, cookies=st.session_state.cookies)
            response_data = response.json()

            if response.status_code == 200 and response_data.get("status") == "success":
                return response_data.get("data")
            else:
                st.error(f"Error: {response_data.get('message')} - {response_data.get('error')}")
        except requests.exceptions.RequestException as e:
            st.error(f"Exception occurred: {e}")
    else:
        st.error("Authentication details not found. Please log in first.")
    return None


# Function to build the AgGrid options
def build_data_grid(df):
    gd = GridOptionsBuilder.from_dataframe(df)
    gd.configure_pagination(enabled=True)
    gd.configure_default_column(groupable=True)
    gd.configure_selection(selection_mode='single', use_checkbox=True)
    gridOptions = gd.build()
    return gridOptions


# Function to delete a record
def delete_record(filename):
    try:
        url = "http://backend:8080/deletefile/"
        params = {"file_name": filename}
        response = requests.post(url, cookies=st.session_state.cookies, params=params)
        result = response.json()

        if response.status_code == 200:
            st.success(f"Record deleted successfully. {result.get('data')}")
            st.rerun()
        else:
            st.error(f"Failed to delete record: {response.json().get('message')}")
    except requests.exceptions.RequestException as e:
        st.error(f"Exception occurred: {e}")


# Main function to display the data
def main():
    st.title("Data Manager")

    raw_data = get_sources()

    # Display message if no data is returned
    if raw_data is None or len(raw_data) == 0:
        st.info("You have no data to display. Please upload files or switch database.")
        return

    # Convert the data to a DataFrame
    if raw_data:
        df = pd.DataFrame(raw_data)
        if df.empty:
            st.error("DataFrame is empty. No data to display.")
            return

        gridOptions = build_data_grid(df)

        try:
            grid_table = AgGrid(
                df,
                height=200,
                gridOptions=gridOptions,
                enable_enterprise_modules=True,
                update_mode=GridUpdateMode.GRID_CHANGED,
                data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
                fit_columns_on_grid_load=True,
                header_checkbox_selection_filtered_only=True,
                use_checkbox=True
            )
        except Exception as e:
            st.error(f"An error occurred while displaying the grid: {e}")
            return

        # Check if there are selected rows
        selected_row = grid_table['selected_rows']
        # Check if any row is selected
        if selected_row is not None:

            if not selected_row.empty:
                if st.button("Delete Selected Record"):
                    filename = selected_row.iloc[0]['fileName']
                    delete_record(filename)
                #if st.button("View Selected Subgraph"):
                #    pass

# Entry point for the script
if __name__ == "__main__":
    main()
