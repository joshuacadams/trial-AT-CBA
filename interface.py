from attr import asdict
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import os
from mdutils.mdutils import MdUtils
import kaleido

import inputs
import outputs
import streamlit_tables as stb
import CBA
import pdf_reports


st.set_page_config(
    page_title = "Active Transport CBA Tool",
    page_icon = "🚴"
    )
    
#Set the page max width
def _max_width_():
    max_width_str = f"max-width: 1000px;"
    st.markdown(
        f"""
    <style>
    .appview-container .main .block-container{{
        {max_width_str}
    }}
    </style>    
    """,
        unsafe_allow_html=True,
    )
_max_width_()

st.title('Active Travel Cost-Benefit Analysis Tool')
st.markdown((open('help_text/sample_intro.txt').read()))

st.header('File')

with st.expander('Save or load projects'):

    uploaded_project = None
    if 'uploaded_project' not in st.session_state:
        st.session_state['uploaded_project'] = False    
    
    if st.button('New Project'):
        st.session_state['uploaded_project'] = False

    #Code for when there are multiple default files
    # default_path_name = st.selectbox('Default parameters',os.listdir('defaults/'))

    default_path_name = 'ATAP.csv'

    if os.path.exists('saved_vars.csv'):
        with open('saved_vars.csv') as file:
            save_button = st.download_button(
                label='Save Project',
                data=file,
                file_name = 'Active Travel Project.csv')

    uploaded_project = st.file_uploader('Upload Saved Project',type='csv')
    if uploaded_project is not None:
        pd.read_csv(uploaded_project).to_csv('uploaded_project.csv')
        st.session_state['uploaded_project'] = True
   
    
    stb.help_button('save_or_load')

#read names and default values from CSVs
inputs.parameter_list = pd.read_csv('names/parameter_list.csv', index_col = 'parameter')
inputs.parameter_list = inputs.parameter_list.astype({
    'min': 'float64',
    'max':'float64',
    'step': 'float64'
    })
inputs.parameter_list = inputs.parameter_list.sort_index()

if st.session_state['uploaded_project'] == True:
    inputs.default_parameters = pd.read_csv(
        'uploaded_project.csv',
        index_col = ['parameter','dimension_0','dimension_1']
        )
    st.markdown('You uploaded a file YAY')
else:
    inputs.default_parameters = pd.read_csv(
        'defaults/'+default_path_name,
        index_col = ['parameter','dimension_0','dimension_1']
        )

inputs.default_parameters = inputs.default_parameters.astype({'value': 'float64'})


inputs.sensitivities = pd.read_csv('names/sensitivity_test.csv')
inputs.sensitivities = inputs.sensitivities.set_index(['sensitivity'])
inputs.sensitivities = inputs.sensitivities.astype({
    'up_min': 'float64',
    'up_max':'float64',
    'up_default':'float64',
    'down_min': 'float64',
    'down_max':'float64',
    'down_default':'float64',    
    'step': 'float64'
    })

# Not sorting the index sometimes gives warnings about indexing speed but also allows variables to be input in the csv order.
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

st.header('Appraisal Settings')

with st.expander('Years and discount rate',False):
    inputs.discount_rate = stb.number_table('discount_rate')
    stb.help_button('discount_rate')

    inputs.appraisal_period = stb.number_table('appraisal_period')
    stb.help_button('appraisal_period')
      
    inputs.start_year = stb.number_table('start_year')    
    stb.help_button('start_year')

    inputs.opening_year = stb.number_table('opening_year')
    stb.help_button('opening_year')
    inputs.opening_year = int(inputs.opening_year)

    inputs.annualisation = stb.number_table('annualisation')
    stb.help_button('annualisation')

st.header('Project Details')

with st.expander('Project Description',False):

    st.subheader('Facility Name')
    
    default_facility_name = inputs.default_parameters.loc['facility_name',np.NaN,np.NaN]['str_value']
    facility_name = st.text_input('Project Name',default_facility_name)
    inputs.facility_name=facility_name
    inputs.saved_vars.loc['facility_name','str_value'] = facility_name

    #Read facility types from the index of relative risk
    facility_type_dict = dict((v,k) for k,v in inputs.default_parameters.loc[('relative_risk','Bicycle'),'name_1'].to_dict().items())

    inputs.facility_length = stb.number_table('facility_length')

    st.subheader('Facility Type')

    default_facility_type_existing = inputs.default_parameters.loc['facility_type_existing',np.NAN,np.NAN]['str_value']
    default_facility_type_new = inputs.default_parameters.loc['facility_type_new',np.NAN,np.NAN]['str_value']

    existing_index = list(facility_type_dict.values()).index(default_facility_type_existing)
    new_index = list(facility_type_dict.values()).index(default_facility_type_new)

    facility_type_existing_text = st.selectbox(
        'Previous facility type',
        list(facility_type_dict.keys()),
        index = existing_index,
        key = 'facility_type_existing_key'
        )
    outputs.exported_inputs['Previous facility type'] = 'Existing type of infrastructure', facility_type_existing_text, None
    
    facility_type_new_text = st.selectbox(
        'New facility type',
        list(facility_type_dict.keys()),
        index = new_index,
        key = 'facility_type_new_key'
        )

    outputs.exported_inputs['New facility type'] = 'Type of infrastructure of project', facility_type_new_text, None

    stb.help_button('facility_type')

    inputs.facility_type_existing = facility_type_dict[facility_type_existing_text]
    inputs.facility_type_new = facility_type_dict[facility_type_new_text]

    inputs.saved_vars.loc['facility_type_existing','str_value'] = inputs.facility_type_existing
    inputs.saved_vars.loc['facility_type_new','str_value'] = inputs.facility_type_new



with st.expander('Project Cost',False):
    #retreive years from year inputs
    inputs.year = list(range(
        int(inputs.start_year),
        int(inputs.start_year) + int(inputs.appraisal_period),
        1
        ))

    inputs.costs = pd.DataFrame(
        index=pd.MultiIndex.from_product([inputs.year,['capital_cost','operating_cost']],names=['year','cost']))

    #read in capex and opex defaults and convert year datatype
    capex_defaults = inputs.default_parameters.loc['capital_cost'][['value']].droplevel('dimension_1')
    capex_defaults.index = capex_defaults.index.astype('float64')

    opex_defaults = inputs.default_parameters.loc['operating_cost'][['value']].droplevel('dimension_1')
    opex_defaults.index = opex_defaults.index.astype('float64')

    #streamlit code
    st.header('Financial costs by year')
    st.subheader('Real capital and operating costs by year')
    stb.help_button('costs')

    cost_input_style = st.radio('Cost input method',('Simple','Table'))

    if cost_input_style == 'Simple':
        st.subheader('Capital cost')
        simple_capex = st.number_input('Total capital cost $',
            value=10000000,
            min_value=0,
            step=1)
        simple_capex_period = st.number_input('Years of construction',
            value=1,
            min_value=1,
            step=1)

        st.subheader('Operating Cost')
        simple_opex = st.number_input('Annual operating cost $',
            value=100000,
            min_value=0,
            step=1)
        simple_opex_escalation = st.number_input('Real growth per year %',
            value=0.0,
            min_value=0.0,
            step=0.1)
        
        for yr in inputs.year:
            if yr <= (inputs.start_year + simple_capex_period - 1):
                inputs.costs.loc[(yr,'capital_cost'),'value'] = simple_capex/simple_capex_period
            else:
                inputs.costs.loc[(yr,'capital_cost'),'value'] = 0
            
            if yr >= inputs.opening_year:
                inputs.costs.loc[(yr,'operating_cost'),'value'] = (simple_opex*
                    (1+(simple_opex_escalation/100))**(yr-inputs.opening_year))
            else:
                inputs.costs.loc[(yr,'operating_cost'),'value'] = 0

        df = inputs.costs.reset_index()
        fig3 = px.bar(df,y='value',x='year',color='cost',orientation='v')
        fig3.update_layout(autosize=True,width=900, title='Costs by year')
        st.plotly_chart(fig3.update_traces(hovertemplate='$%{y:,.0f}'))

       


    if cost_input_style == 'Table':

        colcapex, colopex = st.columns(2)
        colcapex.markdown('Capital cost $')
        colopex.markdown('Operating cost $')
        
        for yr in inputs.year:
            year_number = yr
            #Because appraisal period can change there might not be defaults for later years
            #Check if year is in defaults
            if year_number in capex_defaults.index:
                inputs.costs.loc[(yr,'capital_cost'),'value'] = colcapex.number_input(str(yr),
                    value=int(capex_defaults.loc[year_number,'value']),
                    min_value=0,
                    step=1,
                    key = 'capex'+str(yr)
                    )
            #If year is not in defaults, value defaults to zero
            else:
                inputs.costs.loc[(yr,'capital_cost'),'value'] = colcapex.number_input(str(yr),
                    min_value=0,
                    step=1,
                    key = 'capex'+str(yr)
                    )

            if year_number in opex_defaults.index:
                inputs.costs.loc[(yr,'operating_cost'),'value'] = colopex.number_input(str(yr),
                    value=int(opex_defaults.loc[year_number,'value']),
                    min_value=0,
                    step=1, 
                    key = 'opex'+str(yr)
                    )

            else:
                inputs.costs.loc[(yr,'operating_cost'),'value'] = colopex.number_input(str(yr),
                    min_value=0,
                    step=1, 
                    key = 'opex'+str(yr)
                    )
    
    df = inputs.costs.copy()
    df = df.reset_index()
    df = df.pivot(index='year',columns='cost')
    df.rename(columns={'capital_cost':'Capital','operating_cost':'Operating'},inplace=True)
    outputs.exported_inputs['Costs'] = 'Cost by type', df, '${:,.0f}'

with st.expander('Intersection treatments',False):

   
    inputs.number_of_intersections = stb.number_table('number_of_intersections')
    inputs.number_of_intersections = int(inputs.number_of_intersections)
    if inputs.number_of_intersections > 0:
        intersection_parameters = ['Expected 10-year fatalities','Expected 10-year injuries','Risk reduction %']
        #Need to re-initialise dataframe to change the index each streamlit run
        inputs.intersection_inputs = 0
        inputs.intersection_inputs = pd.DataFrame()
        
        inputs.intersection_inputs.index=pd.MultiIndex.from_product(
            [list(range(inputs.number_of_intersections)),
            intersection_parameters]
            )
        for i in range(inputs.number_of_intersections):
            st.subheader('Intersection '+str(i+1))
            cols = st.columns(3)
            j = 0
            for para in intersection_parameters:
                if ('intersection_inputs',str(i),para) in inputs.default_parameters.index:
                    default=inputs.default_parameters.loc['intersection_inputs',str(i),para]['value']
                else:
                    default=0.0         
                inputs.intersection_inputs.loc[(i,para),'value']=cols[j].number_input(
                    para,
                    min_value=0.0,
                    value=default,
                    key='intersection'+para+str(i)
                    )
                j=j+1
    outputs.exported_inputs['Intersection treatments'] = 'Details of intersection treatments', inputs.intersection_inputs.copy(), None

    
    stb.help_button('intersection_treatments')

st.header('Demand Details')

import demand_calcs
with st.expander('Demand',False):
    
    inputs.base_year_demand = stb.number_table('base_year_demand')
    stb.help_button('base_year_demand')

    inputs.demand_growth = stb.number_table('demand_growth')
    stb.help_button('demand_growth')


    st.header('Customise demand inputs')
    
    blank_demand = pd.DataFrame(0,columns=inputs.base_year_demand.index.tolist(),index=inputs.year)
    blank_demand.index.name = 'year'
    filename = 'demand.csv'
    blank_demand_csv = blank_demand.to_csv(encoding='utf-8', index=True, header=True)
    st.download_button(
        label='Download template demand table',
        data = blank_demand_csv,
        mime='text/csv',
        file_name = 'demand.csv')
    uploaded_demand_csv = st.file_uploader('Click here to upload your completed demand table',type='csv')


    if uploaded_demand_csv is not None:
        raw_uploaded_demand = pd.read_csv(uploaded_demand_csv)
        raw_uploaded_demand.set_index('year', inplace=True)
        raw_uploaded_demand.index.name = 'year'
        st.dataframe(raw_uploaded_demand)
        if raw_uploaded_demand.columns.tolist() != blank_demand.columns.tolist():
            st.error('The rows in the uploaded demand table do not match the template')
        elif raw_uploaded_demand.columns.tolist() != blank_demand.columns.tolist():
            st.error('The columns in the uploaded demand table do not match the template')
        elif raw_uploaded_demand.dtypes.tolist() != blank_demand.dtypes.tolist():
            st.error('There are unexpected characters in the uploaded demand table. Only numbers should be entered.')
        else:
            inputs.custom_demand = True
            st.markdown('✔️ You have succesfully uploaded a demand table. Delete the table to re-enable the number inputs above')
            inputs.uploaded_demand = raw_uploaded_demand.melt(var_name='mode',ignore_index=False).set_index('mode',append=True).swaplevel(i='year',j='mode').sort_index()
    else:
        inputs.custom_demand = False

    basic_demand = demand_calcs.get_basic_demand_frame()

    
    st.header('Assumed demand by mode')
    fig = px.line(basic_demand.rename(columns={'value':'Daily Traffic'}).reset_index(), x='year',y='Daily Traffic', color = "mode")
    fig.update_yaxes(range=[0,round(1.1*max(basic_demand['value']),-2)],hoverformat='.0f')
    st.plotly_chart(fig)


    

with st.expander('Diversion',False):
    inputs.diversion_rate = stb.number_table('diversion_rate',percent=True)
    stb.help_button('diversion_rate')

    inputs.diversion_congestion = stb.number_table('diversion_congestion',percent=True)
    stb.help_button('diversion_congestion')

    inputs.demand_ramp_up = stb.number_table('demand_ramp_up')
    stb.help_button('demand_ramp_up')
    
    demand = demand_calcs.get_demand_frame(basic_demand)
    df = demand.copy()
    df = df.reset_index()
    df = pd.pivot_table(df,'value',index='year',columns='mode',aggfunc='sum')
    for column in df:
        df[column] = df[column].map('{:,.0f}'.format)
    outputs.exported_inputs['Assumed Demand'] = 'Project case demand (including demand ramp up)', df, '{:,.0f}'

st.header('Trip Characteristics')


with st.expander('Trip Characteristics',False):
    inputs.trip_distance_raw = stb.number_table('trip_distance')
    stb.help_button('trip_distance')

    inputs.trip_distance_change = stb.number_table('trip_distance_change')
    stb.help_button('trip_distance_change')

    st.markdown('''---''')

    inputs.surface_distance_prop_base = stb.number_table('surface_distance_prop_base',percent = True)
    inputs.surface_distance_prop_project = stb.number_table('surface_distance_prop_project',percent = True)

    # if inputs.default_parameters.loc['subtract_project_length',np.NaN,np.NaN]['str_value'] == "TRUE":
    #     default_subtract_project_length = True
    # else:
    #     default_subtract_project_length = False
    # inputs.subtract_project_length = st.checkbox(
    #     'Apply infrastructure type proportions only to parts of the trip not on the project infrastructure',
    #     value = default_subtract_project_length
    #     )
    # inputs.saved_vars.loc['subtract_project_length','str_value'] = inputs.subtract_project_length
    
    inputs.subtract_project_length = True

    stb.help_button('surface_distance_prop')

    st.markdown('''---''')

    inputs.time_saving = stb.number_table('time_saving')
    stb.help_button('time_saving')

    inputs.transport_share = stb.number_table('transport_share')
    stb.help_button('transport_share')

st.header('Parameters')

with st.expander('Safety',False):
    inputs.relative_risk = stb.number_table('relative_risk')
    stb.help_button('relative_risk')


with st.expander('Mode attributes',False):
    inputs.speed_active = stb.number_table('speed_active')
    stb.help_button('speed_active')

    # inputs.speed_from_mode = stb.number_table('speed_from_mode')
    # stb.help_button('speed_from_mode')


with st.expander('Unit Values',False):
    inputs.vott = stb.number_table('vott')
    stb.help_button('vott')

    inputs.health_system = stb.number_table('health_system')
    stb.help_button('health_system')

    inputs.health_private = stb.number_table('health_private')
    stb.help_button('health_private')

    inputs.voc_active = stb.number_table('voc_active') 
    # inputs.voc_car = stb.number_table('voc_car')
    # stb.help_button('voc')

    inputs.congestion_cost = stb.number_table('congestion_cost')
    stb.help_button('congestion_cost')

    inputs.crash_cost_active = stb.number_table('crash_cost_active') 
    # inputs.crash_cost_from_mode = stb.number_table('crash_cost_from_mode')
    inputs.injury_cost = stb.number_table('injury_cost')
    stb.help_button('crash_cost')

    inputs.car_externalities = stb.number_table('car_externalities')
    stb.help_button('car_externalities')

    # inputs.road_provision = stb.number_table('road_provision')
    # stb.help_button('road_provision')

    # inputs.parking_cost = stb.number_table('parking_cost')
    # stb.help_button('parking_cost')


#Check if distance travelled on base and project case surfaces is at least as long as facility

# existing_check_distance = (
#     inputs.trip_distance_raw
#     *(inputs.surface_distance_prop_base.loc[inputs.facility_type_existing]/100)
#     )

# new_check_distance = (
#     inputs.trip_distance_raw
#     *(inputs.surface_distance_prop_project.loc[inputs.facility_type_new]/100)
#     )

# distance_check_mode_list = existing_check_distance.index.to_list()
# distance_check_mode_list.remove('Pedestrian')

# for thismode in distance_check_mode_list:
#     if existing_check_distance.loc[thismode,'value'] < inputs.facility_length:
#         st.warning(
#             "Base case distance on "
#             +inputs.facility_type_existing+
#             " for "+thismode+
#             " is shorter than the facility. Assume a higher proportion of trip distance for "
#             +inputs.facility_type_existing+
#             " in the base case"
#             )

# for thismode in distance_check_mode_list:
#     if new_check_distance.loc[thismode,'value'] < inputs.facility_length:
#         st.warning(
#             "Project case distance on "
#             +inputs.facility_type_new+
#             " for "+thismode+
#             " is shorter than the facility. Assume a higher proportion of trip distance for "
#             +inputs.facility_type_new+
#             " in the project case"
#             )

###Append costs to saved vars
# This should remain at the end of the inputs but before any results.
# Whenever a streamlit input is updated, the entire program is re-run with 
# streamlit remembering the new values. 
# stb.number_input and other streamlit code copies the values of all inputs 
# to saved_vars, saved_costs and saved_intersection_inputs which become save_file below.
# This csv is then an input to the save button.
# That way, the inputs to the "previous" run are saved to saved_vars.csv
# and the save button can be at the top of the page
saved_costs = inputs.costs.swaplevel()
saved_costs.sort_index(inplace=True)
saved_costs.index.names = (['parameter','dimension_0'])
saved_costs['dimension_1'] = ""
saved_costs.set_index('dimension_1',inplace=True,append=True)
save_file = inputs.saved_vars.append(saved_costs)

if inputs.number_of_intersections > 0:
    saved_intersection_inputs = inputs.intersection_inputs
    saved_intersection_inputs.insert(loc=0,column = 'parameter',value='intersection_inputs')
    saved_intersection_inputs.set_index('parameter',append=True,inplace=True)
    saved_intersection_inputs.swaplevel(0,2)
    saved_intersection_inputs.index.names = (['parameter','dimension_0','dimension_1'])
    save_file = save_file.append(saved_intersection_inputs)


save_file.to_csv('saved_vars.csv')


col1, col2, col3 = st.columns(3)

calculate_results = col1.checkbox('Calculate results')

if calculate_results:
    calculate_sensitivities = col2.checkbox('Do sensitivity tests')

    if calculate_sensitivities:
        make_downloadables = col3.checkbox('Download results')


    benefits = CBA.calculate_benefits(demand)
    discounted_benefits = CBA.discount_benefits(benefits,inputs.discount_rate)
    discounted_costs = CBA.discount_costs(inputs.costs,inputs.discount_rate)
    outputs.results = CBA.calculate_results(discounted_benefits,discounted_costs)

    outputs.results_dict = {
        'Discounted Benefts':discounted_benefits,
        'Discounted Costs':discounted_costs,
        'Un-discounted Benefits':benefits,
        'Un-discounted Costs':inputs.costs
        }

    outputs.headline_results = [
        ['Present Value of Benefits','$'+"{:,.0f}".format(discounted_benefits.sum().sum())],
        ['Present Value of Costs','$'+"{:,.0f}".format(discounted_costs.sum().sum())],
        ['Net Present Value','$'+"{:,.0f}".format(outputs.results['NPV'])],
        ['Benefit Cost Ratio (BCR1)','{:,.2f}'.format(outputs.results['BCR1'])],
        ['Benefit Cost Ratio (BCR2)','{:,.2f}'.format(outputs.results['BCR2'])],
    ]

    st.header('Results')
    with st.expander('Results',False):
        st.header('Headline results')

        cols = st.columns(3)
        i = 0
        for result in outputs.headline_results:
            cols[i].metric(result[0],value=result[1])
            i = i+1
            if i == 3: i=0

        stb.help_button('headline_results')

        st.header('Breakdown')

        results_format = st.radio('Display',('Charts','Tables'))

        if results_format =='Charts':

            col1,col2,col3 = st.columns(3)

            df_select = col1.selectbox('Measure to chart',outputs.results_dict.keys())
            df = outputs.results_dict[df_select]

            chart_axis = col2.selectbox('Variable 1',df.index.names)
            axis_list_2 = [None] + df.index.names.copy()
            axis_list_2.remove(chart_axis)
            chart_axis_2 = col3.selectbox('Variable 2',axis_list_2)

            if chart_axis_2 is not None:
                df = df.groupby([chart_axis,chart_axis_2]).sum().reset_index()
                fig = px.bar(df,y=chart_axis,x='value',color=chart_axis_2,orientation='h')
                fig.update_layout(autosize=True,width=900, title=df_select)
                st.plotly_chart(fig.update_traces(hovertemplate='$%{x:,.0f}'))
            else:
                df = df.groupby(chart_axis).sum()
                fig = px.bar(df,y=df.index,x='value',orientation='h')
                fig.update_layout(autosize=True,width=900, title=df_select)
                st.plotly_chart(fig.update_traces(hovertemplate='$%{x:,.0f}'))
            
            fig.write_image('Custom chart.svg')
            with open('Custom chart.svg','rb') as file:
                results_download_button = st.download_button(
                    label='Download chart',
                    data=file,
                    file_name = 'Custom chart.svg')
            
        if results_format =='Tables':


            col1,col2,col3 = st.columns(3)
            df_select = col1.selectbox('Measure to chart',outputs.results_dict.keys())
            df = outputs.results_dict[df_select]
            table_vars = df.index.names.copy()

            col2.text('Rows')
            rows = []
            for var in table_vars:
                row = col2.checkbox(var,key=var+'rowkey')
                if row:
                    rows.append(var)
            
            col3.text('Columns')
            column_vars = [x for x in table_vars if x not in rows]
            columns = []
            for var in column_vars:
                column = col3.checkbox(var,key=var+'columnkey')
                if column:
                    columns.append(var)
            
            try:
                df = pd.pivot_table(df,values='value',index=rows,columns=columns,aggfunc='sum',fill_value=0)
            except:
                df = pd.DataFrame(df.sum())
                df.rename(columns={0:'value'},inplace=True)
            df = df.style.format('${:,.0f}')
            df

            with pd.ExcelWriter('Custom table.xlsx') as writer:
                df.to_excel(writer,sheet_name='Custom table')

            with open('Custom table.xlsx','rb') as file:
                results_download_button = st.download_button(
                    label='Download table (Excel)',
                    data=file,
                    file_name = 'Custom table.xlsx')
            

    if calculate_sensitivities:
        with st.expander('Sensitivity testing',False):
            
            st.header('Sensitivity tests')
            col1, col2, col3, col4 = st.columns(4)
            col1.markdown('Sensitivity')
            col2.markdown('NPV')
            col3.markdown('BCR1')
            col4.markdown('BCR2')

            stb.sensitivity_test('discount_rate',bounding_parameter=inputs.discount_rate,convert_to_decimal=False)
            stb.sensitivity_test('capex_sensitivity')
            stb.sensitivity_test('opex_sensitivity')
            stb.sensitivity_test('demand_sensitivity')
            stb.sensitivity_test('trip_distance_sensitivity')
            stb.sensitivity_test('new_trips_sensitivity')
            stb.sensitivity_test('transport_share_sensitivity',convert_to_decimal=False)

            (outputs.exported_sensitivities).set_index('Sensitivity',inplace=True)

            


        if make_downloadables:

            excel_tables = {}

            df = pd.DataFrame.from_dict(outputs.results, orient='index',columns=['value'])
            df = df.append(discounted_benefits.groupby('benefit').sum())
            df = df.append(discounted_costs.groupby('cost').sum())
            df = df.rename(columns={'value':facility_name})
            df.index.name = 'output'
            excel_tables['Headline results'] = df

            excel_tables['Discounted Benefits'] = discounted_benefits.reset_index()
            excel_tables['Discounted Costs'] = discounted_costs.reset_index()
            excel_tables['Undiscounted Benefits'] = benefits.reset_index()
            excel_tables['Undiscounted Costs'] = inputs.costs.reset_index()

            with pd.ExcelWriter(facility_name+' CBA results.xlsx') as writer:
                for sheet in excel_tables:
                    excel_tables[sheet].to_excel(writer,sheet_name=sheet)

            with open(facility_name+' CBA results.xlsx','rb') as file:
                results_download_button = st.download_button(
                    label='Download results (Excel)',
                    data=file,
                    file_name = facility_name+' CBA results.xlsx')
            
            pdf_reports.write_pdf()

            with open(inputs.facility_name+' Cost-Benefit-Analysis.pdf','rb') as file:
                results_download_button = st.download_button(
                    label='Download results (PDF)',
                    data=file,
                    file_name = inputs.facility_name+' Cost-Benefit-Analysis.pdf')