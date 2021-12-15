import numpy as np
import pandas as pd
import inputs
import streamlit as st
import CBA

inputs.saved_vars = pd.DataFrame(columns=['parameter','dimension_0','dimension_1','name_0','name_1','value'])
inputs.saved_vars.set_index(['parameter','dimension_0','dimension_1'],inplace=True)

def parameter_dimensions(parameter):
    if len(inputs.default_parameters.loc[parameter].groupby(level=['dimension_0']).first().index)==0:
        dimensions = 0
    elif len(inputs.default_parameters.loc[parameter].groupby(level=['dimension_1']).first().index)==0:
        dimensions = 1
    else:
        dimensions = 2
    return dimensions

def single_input(parameter, percent=False):
    #pull names and default values
    default = inputs.default_parameters.loc[(parameter,np.nan,np.nan),'value']
    text = inputs.default_parameters.loc[(parameter,np.nan,np.nan),'name_0']
    
    head = inputs.parameter_list.loc[parameter,'name']
    subhead = inputs.parameter_list.loc[parameter,'description']

    minvalue = inputs.parameter_list.loc[parameter,'min']
    maxvalue = inputs.parameter_list.loc[parameter,'max']
    stepvalue = inputs.parameter_list.loc[parameter,'step']

    if stepvalue < .01:
        paraformat = "%.3f"    
    elif stepvalue < .1:
        paraformat = "%.2f"
    elif stepvalue < 1:
        paraformat = "%.1f"
    else:
        paraformat = "%.0f"
    
    #streamlit code
    st.subheader(head)
    #expander_name.subheader(subhead)

    col = st.columns(3)

    widget = col[0].number_input(
        subhead,
        value=default,
        min_value=minvalue,
        max_value=maxvalue,
        step=stepvalue,
        format=paraformat,
        key=parameter
        )
    
    inputs.saved_vars.loc[parameter,'value'] = widget
    inputs.saved_vars.loc[parameter,'name_0'] = text


    return widget

def column_input(parameter, percent=False):
    defaults = inputs.default_parameters.loc[parameter].reset_index(level=1, drop=True)
    default = defaults['value']
    text = defaults['name_0']

    head = inputs.parameter_list.loc[parameter,'name']
    subhead = inputs.parameter_list.loc[parameter,'description']

    minvalue = inputs.parameter_list.loc[parameter,'min']
    maxvalue = inputs.parameter_list.loc[parameter,'max']
    stepvalue = inputs.parameter_list.loc[parameter,'step']

    if stepvalue < .01:
        paraformat = "%.3f"
    elif stepvalue < .1:
        paraformat = "%.2f"
    elif stepvalue < 1:
        paraformat = "%.1f"
    else:
        paraformat = "%.0f"
    
    widget = pd.DataFrame(index = defaults.index)
    widget.index.rename(inputs.parameter_list.loc[parameter,'dimension_0'],inplace=True)

    #streamlit code
    st.subheader(head)
    st.markdown(subhead)

    if len(default) < 4:
        col = st.columns(len(default))
    if len(default) > 3:
        col = st.columns(4)
    
    n=0
    for row in widget.index:
        widget.loc[row,'value'] = col[n].number_input(
            text.loc[row],
            value=default.loc[row],
            min_value=minvalue,
            max_value=maxvalue,
            step=stepvalue,
            format=paraformat,
            key=str(parameter)+"-"+str(row)
            )

        inputs.saved_vars.loc[(parameter,row,""),'value'] = widget.loc[row,'value']
        inputs.saved_vars.loc[(parameter,row,""),'name_0'] = text.loc[row]


        n=n+1
        if n == 4:
            n = 0
    
    if percent==True and round(widget['value'].sum(axis=0),2)!=100:
        st.error("Total must equal 100%, \n Currently " + str(widget['value'].sum(axis=0))+'%')
    
    return widget

def table_input(parameter, percent=False):
    defaults = inputs.default_parameters.loc[parameter]
    default = defaults['value']
    columnnames = inputs.default_parameters.loc[parameter].index.get_level_values('dimension_0').unique().tolist()
    rownames = inputs.default_parameters.loc[parameter].index.get_level_values('dimension_1').unique().tolist()
    
    head = inputs.parameter_list.loc[parameter,'name']
    subhead = inputs.parameter_list.loc[parameter,'description']

    minvalue = inputs.parameter_list.loc[parameter,'min']
    maxvalue = inputs.parameter_list.loc[parameter,'max']
    stepvalue = inputs.parameter_list.loc[parameter,'step']

    if stepvalue < .01:
        paraformat = "%.3f"
    elif stepvalue < .1:
        paraformat = "%.2f"
    elif stepvalue < 1:
        paraformat = "%.1f"
    else:
        paraformat = "%.0f"

    widget = pd.DataFrame(index = defaults.index)
    widget.index.rename(inputs.parameter_list.loc[parameter,['dimension_0','dimension_1']],inplace=True)

    #streamlit code
    st.subheader(head)
    st.markdown(subhead)

    col=st.columns(len(columnnames))
    n=0
    for column in columnnames:
        col[n].markdown(defaults.loc[(column),'name_0'].iloc[0])
        for row in rownames:
            widget.loc[(column,row),'value'] = col[0+n].number_input(
                defaults.loc[(column,row),'name_1'],
                value=default.loc[column,row],
                min_value=minvalue, max_value=maxvalue,
                step=stepvalue, format=paraformat,
                key=str(parameter)+"-"+str(column)+"-"+str(row)
                )
            
            inputs.saved_vars.loc[(parameter,column,row,),'value'] = widget.loc[(column,row),'value']
            inputs.saved_vars.loc[(parameter,column,row),'name_0'] = defaults.loc[(column),'name_0'].iloc[0]
            inputs.saved_vars.loc[(parameter,column,row),'name_1'] = defaults.loc[(column,row),'name_1']

        if percent==True and round(widget.loc[(column)].value.sum(axis=0),2)!=100:
            col[n].error("Total must equal 100%, \n Currently " + str((widget.loc[(column)].value.sum(axis=0)))+'%')
        n=n+1
    return widget
       

def number_table(parameter, percent=False):
    """Return a Pandas DataFrame shaped to match the defaults .csv which is set equal to a streamlit
    number input element."""

    if parameter_dimensions(parameter)==0:
        df = single_input(parameter, percent)
    if parameter_dimensions(parameter)==1:
        df = column_input(parameter, percent)
    if parameter_dimensions(parameter)==2:
        df = table_input(parameter, percent)
    return df

def help_button(parameter):
    """Create a stremlit help button that opens help_text/[parameter].txt in the sidebar"""

    if st.button('Help',key=parameter+' help button'):
        st.sidebar.markdown(open('help_text/'+parameter+'.txt').read())#, unsafe_allow_html=True)

def sensitivity_test(sensitivity,bounding_parameter=None,convert_to_decimal=True):
    """Adds two rows with a streamlit number inputs, takes these inputs to call
    do_sensitivity_CBA and displays the high level CBA results in the row. Default
    values and text are defined in names/sensitivity_test.csv"""

    defaults = inputs.sensitivities.loc[sensitivity]
    cols = st.columns(4)

    if bounding_parameter == None:
        up_min = defaults['up_min']
        down_max = defaults['down_max']
    else:
        up_min = bounding_parameter
        down_max = bounding_parameter

    cols[0].markdown("")
    sens_up= cols[0].number_input(
        min_value=up_min,
        max_value=defaults['up_max'],
        value=defaults['up_default'],
        step=defaults['step'],
        label=defaults['up_name'],
        key=sensitivity+' up',
        format = "%.0f"
        )
    if convert_to_decimal == True:
        sens_up = sens_up/100 + 1
    sens_up_results = CBA.do_sensitivity_CBA(**{sensitivity: sens_up})
    
    # cols[1].markdown('NPV')
    # cols[2].markdown('BCR1')
    # cols[3].markdown('BCR2')
    for i in range(1,4):
        cols[i].markdown("")
    # cols[1].markdown("$""{:,.2f}".format(sens_up_results['NPV']))
    # cols[2].markdown("{:.2f}".format(sens_up_results['BCR1']))
    # cols[3].markdown("{:.2f}".format(sens_up_results['BCR2']))

    cols[1].metric(
        'Net Present Value',
        value='$'+"{:,.0f}".format(sens_up_results['NPV']),
        delta= "{:,.0f}".format(sens_up_results['NPV']-inputs.results['NPV'])
        )
    cols[2].metric(
        'Benefit Cost Ratio (BCR1)',
        value='{:,.2f}'.format(sens_up_results['BCR1']),
        delta='{:,.2f}'.format(sens_up_results['BCR1']-inputs.results['BCR1'])
        )
    cols[3].metric(
        'Benefit Cost Ratio (BCR2)',
        value='{:,.2f}'.format(sens_up_results['BCR2']),
        delta='{:,.2f}'.format(sens_up_results['BCR2']-inputs.results['BCR1'])
        )

    cols[0].markdown("")
    sens_down= cols[0].number_input(
        min_value=defaults['down_min'],
        max_value=down_max,
        value=defaults['down_default'],
        step=defaults['step'],
        label=defaults['down_name'],
        key=sensitivity+' down',
        format = "%.0f"
        )
    if convert_to_decimal == True:
        sens_down = sens_down/100 + 1
    sens_down_results = CBA.do_sensitivity_CBA(**{sensitivity: sens_down})
    # for i in range(1,4):
    #     cols[i].markdown("")
    #     cols[i].markdown("")
    # cols[1].markdown("$""{:,.2f}".format(sens_down_results['NPV']))
    # cols[2].markdown("{:.2f}".format(sens_down_results['BCR1']))
    # cols[3].markdown("{:.2f}".format(sens_down_results['BCR2']))

    cols[1].metric(
        'Net Present Value',
        value='$'+"{:,.0f}".format(sens_down_results['NPV']),
        delta= "{:,.0f}".format(sens_down_results['NPV']-inputs.results['NPV'])
        )
    cols[2].metric(
        'Benefit Cost Ratio (BCR1)',
        value='{:,.2f}'.format(sens_down_results['BCR1']),
        delta='{:,.2f}'.format(sens_down_results['BCR1']-inputs.results['BCR1'])
        )
    cols[3].metric(
        'Benefit Cost Ratio (BCR2)',
        value='{:,.2f}'.format(sens_down_results['BCR2']),
        delta='{:,.2f}'.format(sens_down_results['BCR2']-inputs.results['BCR1'])
        )

    st.markdown('''---''')

