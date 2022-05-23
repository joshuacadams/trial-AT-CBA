import reportlab
import reportlab.rl_config
import reportlab.lib.styles
import pdfdocument.document
import copy
import pandas as pd
import matplotlib
import numpy as np
import PyPDF2
import streamlit as st
import plotly.express as px
from svglib.svglib import svg2rlg

import inputs
import outputs

def df2table(dataframe,style,format='{:,.2f}'):
    """Converts a Pandas DataFrame to a reportlab table. Also drops column levels named None"""
    df = dataframe
    if len(df.columns.names)>1 and None in df.columns.names:
        df = df.droplevel(None,axis=1)
    first_row = [df.index.name] + df.columns.tolist()
    if format is not None:
        try:
            for column in df:
                df[column] = df[column].map(format.format)
        except:
            pass
    array = np.array(df.reset_index())

    table_list = array.tolist()
    table_list.insert(0,first_row)
    table = reportlab.platypus.tables.Table(table_list, style=style)
    return table

table_count = 1
chart_count = 1

#Table style
ts = [('ALIGN', (0, 0), (0, 5), 'LEFT'),
    ('FONT', (0,0), (-1,0), 'Helvetica'),
    ('LINEABOVE',(0,0),(4,0),1,reportlab.lib.colors.black),
    ('LINEBELOW',(0,0),(4,0),0.5,reportlab.lib.colors.black),
    ('FONTSIZE', (1,0), (1, 5), 10),
    ('ALIGN', (1, 0), (4, 30), "RIGHT"),
    ('LINEBELOW',(0,-1),(4,-1),1,reportlab.lib.colors.black)]

PAGE_HEIGHT=reportlab.rl_config.defaultPageSize[1]
PAGE_WIDTH=reportlab.rl_config.defaultPageSize[0]
styles = reportlab.lib.styles.getSampleStyleSheet()
Title = ""
pageinfo = "Headline Results"
def myFirstPage(canvas, doc):
    canvas.saveState() 
    canvas.setFont('Helvetica',10)
    canvas.drawString(97.5 * reportlab.lib.units.mm, 19.05 * reportlab.lib.units.mm, "1")
    canvas.restoreState()
def myLaterPages(canvas, doc):
    canvas.saveState()
    canvas.setFont('Helvetica',10)
    canvas.drawString(97.5 * reportlab.lib.units.mm, 19.05 * reportlab.lib.units.mm, "%d" % (doc.page))
    canvas.restoreState()

class MyPdfDocument(pdfdocument.document.PDFDocument):
    def __init__(self, *args, **kwargs,):
        super(MyPdfDocument, self).__init__(*args, **kwargs)
        self.story = [reportlab.platypus.Spacer(1,0)]
        self.inputs = input

    def generate_style(self, font_name=None, font_size=None):
        super(MyPdfDocument, self).generate_style(font_name, font_size)
        _styles = reportlab.lib.styles.getSampleStyleSheet()
        self.style.normal = copy.deepcopy(_styles['Normal'])
        self.style.normal.alignment = reportlab.lib.enums.TA_CENTER
        self.style.normal.fontName = '%s' % 'Helvetica'
        self.style.normal.fontSize = 17
        self.style.normal.firstLineIndent = 0.4 * reportlab.lib.units.cm
        self.style.normal.spaceBefore = 12.1
        self.style.regular = copy.deepcopy(self.style.normal)
        self.style.regular.fontName = '%s' % 'Helvetica-Bold'
        self.style.regular.alignment = reportlab.lib.enums.TA_LEFT
        self.style.regular.fontSize = 14
        self.style.summary = copy.deepcopy(self.style.regular)
        self.style.summary.fontSize = 12
        self.style.table = copy.deepcopy(self.style.normal)
        self.style.table.fontSize = 10
        self.style.plot = copy.deepcopy(self.style.regular)
        self.style.plot.fontSize = 10

    def image(self, image_path, width=19 * reportlab.lib.units.cm, style=None):
        img = matplotlib.image.imread(image_path)
        x, y = img.shape[:2]
        image = reportlab.platypus.Image(image_path, width=width, height=width * x / y)
        self.story.append(image)

    def drawing(self,image_path,title):
        self.this_chart = [reportlab.platypus.Spacer(1,0)]
        global chart_count
        text = "Chart "+str(chart_count)+": "+title
        chart_count = chart_count +1

        self.this_chart.append(reportlab.platypus.Paragraph(text, self.style.table))
        self.this_chart.append(reportlab.platypus.Spacer(1,0.133333*reportlab.lib.units.inch))
        
        drawing = svg2rlg(image_path)
        self.this_chart.append(drawing)
        self.this_chart.append(reportlab.platypus.Spacer(1,0.133333*reportlab.lib.units.inch))
        self.this_chart = reportlab.platypus.KeepTogether(self.this_chart)
        
        
        self.story.append(self.this_chart)

    
    
    def together_table(self,heading,dataframe,style=ts,format='{:,.2f}',column_names=None,big_head=None):
        self.this_table = [reportlab.platypus.Spacer(1,0)]
        global table_count
        if big_head is not None:
            self.this_table.append(reportlab.platypus.Paragraph(big_head, self.style.summary))
        text = "Table "+str(table_count)+": "+heading
        self.inputs = input
        self.this_table.append(reportlab.platypus.Paragraph(text, self.style.table))
        self.this_table.append(reportlab.platypus.Spacer(1,0.133333*reportlab.lib.units.inch))
        if column_names is not None:
            if len(dataframe.columns) == len(column_names):
                dataframe.columns = column_names
        table = df2table(dataframe,style,format)
        self.this_table.append(table)
        table_count = table_count + 1
        self.this_table.append(reportlab.platypus.Spacer(1,0.3125*reportlab.lib.units.inch))
        self.this_table = reportlab.platypus.KeepTogether(self.this_table)
        self.story.append(self.this_table)
    

    def result(self):

        # Create document and title
        doc = reportlab.platypus.SimpleDocTemplate(filename=inputs.facility_name+' Cost-Benefit-Analysis.pdf', title=inputs.facility_name+' Cost-Benefit-Analysis')
        resulttext = inputs.facility_name+' Cost-Benefit-Analysis'
        self.story.append(reportlab.platypus.Paragraph(resulttext, self.style.normal))
        self.story.append(reportlab.platypus.Spacer(1,0.3125*reportlab.lib.units.inch))
        
        global table_count
        table_count = 1
        global chart_count
        chart_count = 1


        resulttext = "Outputs"         
        self.story.append(reportlab.platypus.Paragraph(resulttext, self.style.regular))
        self.story.append(reportlab.platypus.Spacer(1,121/720*reportlab.lib.units.inch))
        resulttext = "Summary"
        self.story.append(reportlab.platypus.Paragraph(resulttext, self.style.summary))
        self.story.append(reportlab.platypus.Spacer(1,0.2*reportlab.lib.units.inch))

        resulttext = "Table 1: Headline Results"
        self.story.append(reportlab.platypus.Paragraph(resulttext, self.style.table))
        self.story.append(reportlab.platypus.Spacer(1,1/9*reportlab.lib.units.inch))
        table_list = outputs.headline_results.copy()
        table_list.insert(0,['Metric',''])
        table = reportlab.platypus.tables.Table(table_list, style=ts)
        self.story.append(table)
        self.story.append(reportlab.platypus.Spacer(1,0.3125*reportlab.lib.units.inch))
        
        table_count = table_count +1
        # # Present value tables

        resulttext = "Present value of benefits"
        self.story.append(reportlab.platypus.Paragraph(resulttext, self.style.summary))
        self.story.append(reportlab.platypus.Spacer(1,0.2*reportlab.lib.units.inch))

        discounted_benefits = outputs.results_dict['Discounted Benefts'].copy()

        pv_tables = [
            ['Benefits by Benefit Type',discounted_benefits.groupby('benefit').sum()],
            ['Benefits by Mode', discounted_benefits.groupby('mode').sum()],
            ['Benefits by Mode and diversion source',discounted_benefits.groupby(['mode','from_mode']).sum().unstack()],
            ['Benefits by mode and benefit type',discounted_benefits.groupby(['benefit','mode']).sum().unstack()],
            ['Benefits by year',discounted_benefits.groupby('year').sum()]            
        ]   
        
        for table in pv_tables:
            self.together_table(table[0],table[1],format='${:,.0f}',column_names=['Present Value'])

        resulttext = "Sensitivity Tests"
        self.story.append(reportlab.platypus.Paragraph(resulttext, self.style.summary))
        self.story.append(reportlab.platypus.Spacer(1,0.2*reportlab.lib.units.inch))
        self.together_table('Sensitivity Tests',outputs.exported_sensitivities,format=None)

        self.story.append(reportlab.platypus.PageBreak())
        
        resulttext = 'Benefit charts'
        self.story.append(reportlab.platypus.Paragraph(resulttext, self.style.summary))
        self.story.append(reportlab.platypus.Spacer(1,0.2*reportlab.lib.units.inch))

        margin = {'l': 0,'r': 0, 'b': 0, 't': 0,'pad': 0}
        def fig_format():
            return fig.update_layout(
                margin = {'l': 0,'r': 0, 'b': 0, 't': 0,'pad': 0},
                font_family='Helvetica',
                font_size=8,
                font_color='black',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )


        df = discounted_benefits.groupby('mode').sum()
        fig = px.bar(df,y=df.index,x='value',orientation='h')
        fig_format()
        fig.update_layout(height=200,width=450)
        fig.write_image("images/fig.svg")
        self.drawing("images/fig.svg",'Benefits by mode')

        df = discounted_benefits.groupby('benefit').sum()
        fig = px.bar(df,y=df.index,x='value',orientation='h')
        fig_format()
        fig.update_layout(height=200,width=450)
        fig.write_image("images/fig.svg")
        self.drawing("images/fig.svg",'Benefits by benefit type')

        df = discounted_benefits.groupby('year').sum()
        fig = px.bar(df,y='value',x=df.index,orientation='v')
        fig_format()
        fig.update_layout(height=200,width=450,margin=margin)
        fig.write_image("images/fig.svg")
        self.drawing("images/fig.svg",'Benefits by year')
        
        df = discounted_benefits.groupby(['benefit','mode']).sum().reset_index()
        fig = px.bar(df,y='mode',x='value',color='benefit',orientation='h')
        fig_format()
        fig.update_layout(height=350,width=450,margin=margin)
        fig.update_layout(legend=dict(
            x=0.7,
            y=1,
            traceorder='normal'))
        fig.write_image("images/fig.svg")
        self.drawing("images/fig.svg",'Benefits by mode and benefit type')


        self.story.append(reportlab.platypus.PageBreak())

        # Add heading and discount rate
        resulttext = 'Inputs'
        self.story.append(reportlab.platypus.Paragraph(resulttext, self.style.regular))
        self.story.append(reportlab.platypus.Spacer(1,0.133333*reportlab.lib.units.inch))

        paras = outputs.exported_inputs.copy()

        for para in paras:
            if type(paras[para][1]) == pd.core.frame.DataFrame:
                self.together_table(paras[para][0],paras[para][1],format=paras[para][2],big_head=para)
            else:
                resulttext = para
                self.story.append(reportlab.platypus.Paragraph(resulttext, self.style.summary))
                text = paras[para][0]
                self.story.append(reportlab.platypus.Paragraph(text, self.style.table))
                try:
                    text = paras[para][2].format(paras[para][1])
                except:
                    text = str(paras[para][1])
                self.story.append(reportlab.platypus.Paragraph(text, self.style.table))




        # resulttext = 'Discount Rate: '+"{:,.0f}%".format(inputs.discount_rate)
        # self.story.append(reportlab.platypus.Paragraph(resulttext, self.style.table))
        # self.story.append(reportlab.platypus.Spacer(1,0.133333*reportlab.lib.units.inch))

        # # Create table for year parameters
        # resulttext = 'Table 2: Year Parameters'
        # self.story.append(reportlab.platypus.Paragraph(resulttext, self.style.table))
        # self.story.append(reportlab.platypus.Spacer(1,0.133333*reportlab.lib.units.inch))
        # total = {'Parameter': ['Appraisal period', 'Start year', 'Opening year', 'Annualisation no.'],'Value':["{:,.0f}".format(inputs.appraisal_period), "{:.0f}".format(inputs.start_year), "{:.0f}".format(inputs.opening_year), "{:.0f}".format(inputs.annualisation)], 'Unit' : ['years', '', '', '']}
        # total = pd.DataFrame(total)
        # lista = [total.columns[:,].values.astype(str).tolist()] + total.values.tolist()
        # table = reportlab.platypus.tables.Table(lista, style=ts)

        # # Append table
        # self.story.append(table)
        # self.story.append(reportlab.platypus.Spacer(1,0.3125*reportlab.lib.units.inch))

        # # Create table for project descriptors
        # resulttext = 'Table 3: Project Description'
        # self.story.append(reportlab.platypus.Paragraph(resulttext, self.style.table))
        # self.story.append(reportlab.platypus.Spacer(1,0.133333*reportlab.lib.units.inch))
        # total = {'Parameter': ['Facility Length', 'Previous facility type', 'New facility type'],'Value':["{:.1f}".format(inputs.facility_length), inputs.facility_type_existing.replace('on-road', 'On-road (no provision)').replace('bikelane', 'On-road Bicycle lane').replace('bikeway', 'Off-road path').replace('footpath', 'Footpath'), inputs.facility_type_new.replace('on-road', 'On-road (no provision)').replace('bikelane', 'On-road Bicycle lane').replace('bikeway', 'Off-road path').replace('footpath', 'Footpath')], 'Unit' : ['km', '', '']}
        # total = pd.DataFrame(total)
        # lista = [total.columns[:,].values.astype(str).tolist()] + total.values.tolist()
        # table = reportlab.platypus.tables.Table(lista, style=ts)

        # # Append table 
        # self.story.append(table)
        # self.story.append(reportlab.platypus.Spacer(1,0.3125*reportlab.lib.units.inch))

        # # Create heading for project cost table
        # resulttext = 'Table 4: Project Cost'
        # self.story.append(reportlab.platypus.Paragraph(resulttext, self.style.table))
        # self.story.append(reportlab.platypus.Spacer(1,0.133333*reportlab.lib.units.inch))

        # # Reset index for input costs, pivot on cost, reset index and format years
        # total = inputs.costs.reset_index()
        # total = total.pivot(index="year", columns="cost", values="value")
        # total.reset_index(inplace=True)
        # total['year'] = ['{:.0f}'.format(x) for x in total['year']]

        # # Format capital cost, format operating cost, change columns and create table
        # total['capital_cost'] = ['${:,.0f}'.format(x) for x in total['capital_cost']]
        # total['operating_cost'] = ['${:,.0f}'.format(x) for x in total['operating_cost']]
        # total.columns=['Year', 'Cap. Ex.', 'Op. Ex.']
        # lista = [total.columns[:,].values.astype(str).tolist()] + total.values.tolist()

        # # Create layout for table, create table, add to document and add space
        # ts1 = [('ALIGN', (1,1), (-1,-1), 'CENTER'),
        #          ('FONT', (0,0), (-1,0), 'Helvetica'),
        #          ('LINEABOVE',(0,0),(4,0),1,reportlab.lib.colors.black),
        #          ('LINEBELOW',(0,0),(4,0),0.5,reportlab.lib.colors.black),
        #          ('FONTSIZE', (1,0), (1, 5), 10),
        #          ('ALIGN', (1, 0), (4, 61), "RIGHT"),
        #          ('LINEBELOW',(0,-1),(4,-1),1,reportlab.lib.colors.black)]
        # table = reportlab.platypus.tables.Table(lista, style=ts1, repeatRows=1)
        # self.story.append(table)
        # self.story.append(reportlab.platypus.PageBreak())

        # # Add subheading
        # resulttext = 'Demand' 
        # self.story.append(reportlab.platypus.Paragraph(resulttext, self.style.summary))
        # self.story.append(reportlab.platypus.Spacer(1,0.133333*reportlab.lib.units.inch))

        # # Create table for demand
        # resulttext = 'Table 5: Demand'
        # self.story.append(reportlab.platypus.Paragraph(resulttext, self.style.table))
        # self.story.append(reportlab.platypus.Spacer(1,0.133333*reportlab.lib.units.inch))
        # total = {'Parameter': ['Base year demand - Bicycle', 'Base year demand - Pedestrian', 'Base year demand - e-Bike', 'Base year demand - e-Scooter'],'Value': ['{:,.0f}'.format(x) for x in inputs.base_year_demand['value']], 'Unit' : ['', '', '', '']}
        # total = pd.DataFrame(total)

        # # If there isn't custom demand:
        # if inputs.custom_demand == False:

        #     # Create dataframe of demand growth rates and append
        #     table =  {'Parameter': ['Demand growth - Bicycle', 'Demand growth - Pedestrian', 'Demand growth - e-Bike', 'Demand growth - e-Scooter'],'Value': inputs.demand_growth['value'], 'Unit' : ['%', '%', '%', '%']}
        #     table = pd.DataFrame(table)
        #     total = total.append(table)

        # # Append table
        # lista = [total.columns[:,].values.astype(str).tolist()] + total.values.tolist()
        # table = reportlab.platypus.tables.Table(lista, style=ts)
        # self.story.append(table)
        # self.story.append(reportlab.platypus.Spacer(1,0.3125*reportlab.lib.units.inch))

        # # Create table for demand escalation
        # resulttext = 'Table 6: Demand escalation'
        # self.story.append(reportlab.platypus.Paragraph(resulttext, self.style.table))
        # self.story.append(reportlab.platypus.Spacer(1,0.133333*reportlab.lib.units.inch))
        # total = basic.unstack(level=0)
        # total.reset_index(inplace=True)
        # total.columns = ['Year', 'Cyclists', 'Pedestrians', 'e-Bike users', 'e-Scooter users']
        # total = total.astype(int)

        # # Append table
        # lista = [total.columns[:,].values.astype(str).tolist()] + total.values.tolist()
        # table = reportlab.platypus.tables.Table(lista, style=ts1)
        # self.story.append(table)

        # # Add plot, page break and subheading
        # self.image('filename5.png')
        # self.story.append(reportlab.platypus.PageBreak())
        # resulttext = 'Diversion'
        # self.story.append(reportlab.platypus.Paragraph(resulttext, self.style.summary))
        # self.story.append(reportlab.platypus.Spacer(1,0.133333*reportlab.lib.units.inch))

        # # Create table for diversion rates
        # resulttext = 'Table 7: Diversion rates'
        # self.story.append(reportlab.platypus.Paragraph(resulttext, self.style.table))
        # self.story.append(reportlab.platypus.Spacer(1,0.133333*reportlab.lib.units.inch))
        # total = {'Parameters': ['Car to bicycle', 'PT to bicycle', 'Reassign bicycle', 'New trips bicycle', 'Car to walk', 'PT to walk', 'Reassign walk', 'New trips walk', 'Car to e-Bike', 'PT to e-Bike', 'Reassign e-Bike', 'New trips e-Bike', 'Car to e-Scooter', 'PT to e-Scooter', 'Reassign e-Scooter', 'New trips e-Scooter'], 'Value': inputs.diversion_rate['value'], 'Unit':['%','%','%','%','%','%','%','%','%','%','%','%','%','%','%','%']}
        # total = pd.DataFrame(total)

        # # Append table
        # lista = [total.columns[:,].values.astype(str).tolist()] + total.values.tolist()
        # table = reportlab.platypus.tables.Table(lista, style=ts1)
        # self.story.append(table)
        # self.story.append(reportlab.platypus.Spacer(1,0.2*reportlab.lib.units.inch))

        # # Create table for proportion of diversion
        # resulttext = "Table 8: Proportion of diversion from motor vehicle by congestion period:"
        # self.story.append(reportlab.platypus.Paragraph(resulttext, self.style.table))
        # self.story.append(reportlab.platypus.Spacer(1,0.2*reportlab.lib.units.inch))
        # total = divert
        # total['value'] = ['{:.0f}%'.format(x) for x in total['value']]
        # total = total.T
        # total.columns=['Busy', 'Medium', 'Light']

        # # Format table for reportlab and create new table style
        # lista = [total.columns[:,].values.astype(str).tolist()] + total.values.tolist()
        # ts2 = [('FONT', (0,0), (-1,0), 'Helvetica'),
        #          ('LINEABOVE',(0,0),(3,0),1,reportlab.lib.colors.black),
        #          ('LINEBELOW',(0,0),(3,0),0.5,reportlab.lib.colors.black),
        #          ('FONTSIZE', (1,0), (1, 5), 10),
        #          ('ALIGN', (0, 1), (3, 8), "RIGHT"),
        #          ('LINEBELOW',(0,-1),(3,-1),1,reportlab.lib.colors.black)]

        # # Append table
        # table = reportlab.platypus.tables.Table(lista, style=ts2)
        # self.story.append(table)
        # self.story.append(reportlab.platypus.Spacer(1,0.2*reportlab.lib.units.inch))

        # # Add demand ramp up period
        # resulttext = 'Demand ramp up period: ' + '{:.0f} years'.format(inputs.demand_ramp_up)
        # self.story.append(reportlab.platypus.Paragraph(resulttext, self.style.table))
        # self.story.append(reportlab.platypus.PageBreak())

        # # Add subheading
        # resulttext = "Demand Attributes"
        # self.story.append(reportlab.platypus.Paragraph(resulttext, self.style.summary))
        # self.story.append(reportlab.platypus.Spacer(1,0.2*reportlab.lib.units.inch))

        # # Create table for average speed for active modes
        # resulttext = "Table 9: Average speed for active modes by facility type"
        # self.story.append(reportlab.platypus.Paragraph(resulttext, self.style.table))
        # self.story.append(reportlab.platypus.Spacer(1,0.2*reportlab.lib.units.inch))
        # total = inputs.speed_active.copy()
        # total.reset_index(inplace=True)
        # total = total.pivot(index="surface", columns="mode", values="value")
        # total.reset_index(inplace=True)
        # total['surface'] = ['On-road Bicycle lane',  'Off-road path', 'Footpath', 'On-road (no provision)']
        # total.rename(columns={'surface':'Surface'}, inplace=True)
        # total = pd.DataFrame(total)
            
        # # Append table
        # lista = [total.columns[:,].values.astype(str).tolist()] + total.values.tolist()
        # table = reportlab.platypus.tables.Table(lista, style=ts1)
        # self.story.append(table)
        # self.story.append(reportlab.platypus.Spacer(1,0.2*reportlab.lib.units.inch))

        # # Create table for average speed for other modes
        # resulttext = "Table 10: Average speed for other modes"
        # self.story.append(reportlab.platypus.Paragraph(resulttext, self.style.table))
        # self.story.append(reportlab.platypus.Spacer(1,0.2*reportlab.lib.units.inch))
        # total = inputs.speed_from_mode.copy()
        # if [ col  for col, dt in total.dtypes.items() if dt == object] != ['value']:
        #     total['value'] = ['{:.0f} km/h'.format(x) for x in total['value']]
        # total = total.T
        # total.columns=['Motor vehicle', 'Transit']

        # # Append table
        # lista = [total.columns[:,].values.astype(str).tolist()] + total.values.tolist()
        # table = reportlab.platypus.tables.Table(lista, style=ts2)
        # self.story.append(table)
        # self.story.append(reportlab.platypus.Spacer(1,0.2*reportlab.lib.units.inch))

        # # Add subheading
        # resulttext = "Trip Characteristics"
        # self.story.append(reportlab.platypus.Paragraph(resulttext, self.style.summary))
        # self.story.append(reportlab.platypus.Spacer(1,0.2*reportlab.lib.units.inch))

        # # Create table for trip distances
        # resulttext = "Table 11: Trip Distances"
        # self.story.append(reportlab.platypus.Paragraph(resulttext, self.style.table))
        # self.story.append(reportlab.platypus.Spacer(1,0.2*reportlab.lib.units.inch))
        # total = {'Parameter': ['Average distance: bicycle', 'Average distance: walk', 'Average distance: e-Bike', 'Average distance: e-Scooter'], 'Value': inputs.trip_distance_raw['value'], 'Unit':['km','km','km','km']}
        # total = pd.DataFrame(total)
        # table = {'Parameter': ['Change in distance: bicycle', 'Change in distance: walk', 'Change in distance: e-Bike', 'Change distance: e-Scooter'], 'Value': inputs.trip_distance_change['value'], 'Unit':['km','km','km','km']}
        # table = pd.DataFrame(table)
        # total = total.append(table)

        # # Append table
        # lista = [total.columns[:,].values.astype(str).tolist()] + total.values.tolist()
        # table = reportlab.platypus.tables.Table(lista, style=ts1, repeatRows=1)
        # self.story.append(table)
        # self.story.append(reportlab.platypus.Spacer(1,0.2*reportlab.lib.units.inch))

        # # Create table for proportion of trip distaces by surface in base case
        # resulttext = "Table 13: Proportion of trip distance by surface (in the base case):"
        # self.story.append(reportlab.platypus.Paragraph(resulttext, self.style.table))
        # self.story.append(reportlab.platypus.Spacer(1,0.2*reportlab.lib.units.inch))
        # total = inputs.surface_distance_prop_base.copy()
        # total = total.T
        # total.iloc[0] = ['{:.0f}%'.format(x) for x in total.iloc[0]]
        # total.columns=['On-road (no provision)', 'On-road Bicycle lane',  'Off-road path', 'Footpath']
            
        # # Append table
        # lista = [total.columns[:,].values.astype(str).tolist()] + total.values.tolist()
        # table = reportlab.platypus.tables.Table(lista, style=ts2)
        # self.story.append(table)
        # self.story.append(reportlab.platypus.Spacer(1,0.2*reportlab.lib.units.inch))
            
        # # Create table for proportion of trip distaces by surface in project case
        # resulttext = "Table 14: Proportion of trip distance by surface (in the project case):"
        # self.story.append(reportlab.platypus.Paragraph(resulttext, self.style.table))
        # self.story.append(reportlab.platypus.Spacer(1,0.2*reportlab.lib.units.inch))
        # total = inputs.surface_distance_prop_project
        # if [ col  for col, dt in total.dtypes.items() if dt == object] != ['value']:
        #     total['value'] = ['{:.0f}%'.format(x) for x in total['value']]
        # total = total.T
        # total.columns=['On-road (no provision)', 'On-road Bicycle lane',  'Off-road path', 'Footpath']
            
        # # Append table
        # lista = [total.columns[:,].values.astype(str).tolist()] + total.values.tolist()
        # table = reportlab.platypus.tables.Table(lista, style=ts2)
        # self.story.append(table)
        # self.story.append(reportlab.platypus.Spacer(1,0.2*reportlab.lib.units.inch))

        # # Create table for change in travel time for intersection treatments
        # resulttext = "Table 15: Change in travel time from intersection treatments (in seconds):"
        # self.story.append(reportlab.platypus.Paragraph(resulttext, self.style.table))
        # self.story.append(reportlab.platypus.Spacer(1,0.2*reportlab.lib.units.inch))
        # total = inputs.time_saving.copy()
        # total['value'] = ['{:.0f}'.format(x) for x in total['value']]
        # total = total.T

        # # Append table
        # lista = [total.columns[:,].values.astype(str).tolist()] + total.values.tolist()
        # table = reportlab.platypus.tables.Table(lista, style=ts2)
        # self.story.append(table)
        # self.story.append(reportlab.platypus.Spacer(1,0.2*reportlab.lib.units.inch))

        # # Create table for proportion of trips for transport purposes
        # resulttext = "Table 16: Proportion of trips for transport purposes:"
        # self.story.append(reportlab.platypus.Paragraph(resulttext, self.style.table))
        # self.story.append(reportlab.platypus.Spacer(1,0.2*reportlab.lib.units.inch))
        # total = inputs.transport_share.copy()
        # total['value'] = ['{:.0f}%'.format(x) for x in total['value']]
        # total = total.T

        # # Append table
        # lista = [total.columns[:,].values.astype(str).tolist()] + total.values.tolist()
        # table = reportlab.platypus.tables.Table(lista, style=ts2)
        # self.story.append(table)
        # self.story.append(reportlab.platypus.Spacer(1,0.2*reportlab.lib.units.inch))

        # # Add subheading
        # resulttext = "Safety"
        # self.story.append(reportlab.platypus.Paragraph(resulttext, self.style.summary))
        # self.story.append(reportlab.platypus.Spacer(1,0.2*reportlab.lib.units.inch))

        # # Create table for relative risk
        # resulttext = "Table 17: Relative risk compared to on-road (on-road = 1):"
        # self.story.append(reportlab.platypus.Paragraph(resulttext, self.style.table))
        # self.story.append(reportlab.platypus.Spacer(1,0.2*reportlab.lib.units.inch))
        # total = inputs.relative_risk.copy()
        # total.reset_index(inplace=True)
        # total = total.pivot(index="surface", columns="mode", values="value")
        # total.reset_index(inplace=True)
        # total['surface'] = ['On-road Bicycle lane',  'Off-road path', 'Footpath', 'On-road (no provision)']
        # total.rename(columns={'surface':'Surface'}, inplace=True)

        # # Append table
        # lista = [total.columns[:,].values.astype(str).tolist()] + total.values.tolist()
        # table = reportlab.platypus.tables.Table(lista, style=ts)
        # self.story.append(table)
        # self.story.append(reportlab.platypus.PageBreak())

        # # Add subheading
        # resulttext = "Intersection Treatments"
        # self.story.append(reportlab.platypus.Paragraph(resulttext, self.style.summary))
        # self.story.append(reportlab.platypus.Spacer(1,0.2*reportlab.lib.units.inch))

        # # Add number of intersection treatments
        # resulttext = "Number of intersection treatments: {}".format(inputs.number_of_intersections)
        # self.story.append(reportlab.platypus.Paragraph(resulttext, self.style.table))
        # self.story.append(reportlab.platypus.Spacer(1,0.2*reportlab.lib.units.inch))

        # if inputs.intersection_inputs.empty:
        #     pass
        # else:
        #     # Get intersection values, format, reset index and pivot dataframe
        #     total = inputs.intersection_inputs.copy()
        #     total['value'] = ['{:.2f}'.format(x) for x in total['value']]
        #     total.reset_index(inplace=True)
        #     total = total.pivot(index="level_0", columns="level_1", values="value")
            
        #     # For each intersection:
        #     for i in range(inputs.number_of_intersections):
                
        #         # Create table for intersection
        #         resulttext = "Table {}: Intersection {}:".format(18 + i, i + 1)
        #         self.story.append(reportlab.platypus.Paragraph(resulttext, self.style.table))
        #         self.story.append(reportlab.platypus.Spacer(1,0.2*reportlab.lib.units.inch))
        #         total1 = pd.DataFrame(total.iloc[i]).T
        #         total1.columns = ["Expected 10-year fatalities", "Expected 10-year injuries", "Risk reduction"]
        #         total1["Risk reduction"] += "%"
        #         lista = [total1.columns[:,].values.astype(str).tolist()] + total1.values.tolist()
        #         table = reportlab.platypus.tables.Table(lista, style=ts2)
                
        #         # Append table
        #         self.story.append(table)
        #         self.story.append(reportlab.platypus.Spacer(1,0.2*reportlab.lib.units.inch))
            
        # # Add pagebreak and subheading
        # self.story.append(reportlab.platypus.PageBreak())
        # resulttext = "Unit Values"
        # self.story.append(reportlab.platypus.Paragraph(resulttext, self.style.summary))
        # self.story.append(reportlab.platypus.Spacer(1,0.2*reportlab.lib.units.inch))

        # # Create table for units values with health values
        # resulttext = "Table {}: Unit values".format(18 + inputs.number_of_intersections)
        # self.story.append(reportlab.platypus.Paragraph(resulttext, self.style.table))
        # self.story.append(reportlab.platypus.Spacer(1,0.2*reportlab.lib.units.inch))
        # total = {'Parameter': ['Health: Cycling', 'Health: Walking', 'Health: e-Bike riding', 'Health: e-Scooter riding'], 'Value': ["${:.2f}".format(x) for x in inputs.health_system['value']]}
        # total = pd.DataFrame(total)

        # # Append decongestion values
        # table = {'Parameter': ['Decongestion: Busy', 'Decongestion: Medium', 'Decongestion: Light'], 'Value': ["${:.2f}".format(x) for x in inputs.congestion_cost['value']]}        
        # table = pd.DataFrame(table)
        # total = total.append(table)

        # # Append vehicle operating costs
        # table = {'Parameter' : ['Vehicle operating costs'], 'Value': "${:.2f}".format(inputs.voc_car)}
        # table = pd.DataFrame(table)
        # total = total.append(table)

        # # Append active travel costs
        # table = {'Parameter' : ['Bicycle operating costs', 'Pedestrian costs', 'e-Bike operating costs', 'e-Scooter operating costs'], 'Value': ["${:.2f}".format(x) for x in inputs.voc_active['value']]}
        # table = pd.DataFrame(table)
        # total = total.append(table)

        # # Append car externalities
        # table = {'Parameter' : ['Noise reduction', 'Air quality', 'Greenhouse gas emissions'], 'Value': ["${:.2f}".format(x) for x in inputs.car_externalities['value']]}
        # table = pd.DataFrame(table)
        # total = total.append(table)

        # # Append infrastructure savings
        # table = {'Parameter' : ['Infra. (roadway) savings'], 'Value': "${:.2f}".format(inputs.road_provision)}
        # table = pd.DataFrame(table)
        # total = total.append(table)

        # # Append parking cost savings
        # table = {'Parameter' : ['Parking cost savings'], 'Value': "${:.2f}".format(inputs.parking_cost)}
        # table = pd.DataFrame(table)
        # total = total.append(table)
            
        # # Append morbidity and mortalitity costs avoided
        # table = {'Parameter' : ['Morbidity and Mortality avoided: Bicycle', 'Morbidity and Mortality avoided: Pedestrian', 'Morbidity and Mortality avoided: e-Bike', 'Morbidity and Mortality avoided: e-Scooter'], 'Value': ["${:.2f}".format(x) for x in inputs.health_private['value']]}
        # table = pd.DataFrame(table)
        # total = total.append(table)

        # # Append crash costs for active travel
        # table = {'Parameter' : ['Crash cost: Bicycle', 'Crash cost: Pedestrian', 'Crash cost: e-Bike', 'Crash cost: e-Scooter'], 'Value': ["${:.2f}".format(x) for x in inputs.crash_cost_active['value']]}
        # table = pd.DataFrame(table)
        # total = total.append(table)

        # # Append crash costs for other modes
        # table = {'Parameter' : ['Crash cost: Motor vehicle', 'Crash cost: Transit'], 'Value': ["${:.2f}".format(x) for x in inputs.crash_cost_from_mode['value']]}
        # table = pd.DataFrame(table)
        # total = total.append(table)

        # # Append value of travel time savings
        # table = {'Parameter' : ['Value of travel time savings'], 'Value': "${:.2f}".format(inputs.vott)}
        # table = pd.DataFrame(table)
        # total = total.append(table)

        # # Append costs of fatalities and injuries
        # table = {'Parameter' : ['Cost per fatality', 'Cost per injury'], 'Value': ["${:.0f}".format(x) for x in inputs.injury_cost['value']]}
        # table = pd.DataFrame(table)
        # total = total.append(table)

        # # Append table and page break
        # lista = [total.columns[:,].values.astype(str).tolist()] + total.values.tolist()
        # table = reportlab.platypus.tables.Table(lista, style=ts)
        # self.story.append(table)
        # self.story.append(reportlab.platypus.PageBreak())

        # # Add heading
        # resulttext = "Sensitivity Tests"
        # self.story.append(reportlab.platypus.Paragraph(resulttext, self.style.normal))
        # self.story.append(reportlab.platypus.Spacer(1,0.2*reportlab.lib.units.inch))

        # total = np.array(['Central Assesment', inputs.results['NPV'], inputs.results['BCR1'], inputs.results['BCR2']])
        # total = pd.DataFrame(total)
        # total = total.T
        # total.columns = ['', 'NPV $000', 'BCR1', 'BCR2']
            
        # # Create table of high discount rate sensitivity
        # table = discount[0]
        # if [ col  for col, dt in table.dtypes.items() if dt == object] != ['']:
        #     table[''] = ["High discount rate ({:.0f}%)".format(x) for x in table['']]
        # total = total.append(table)

        # # Create table of low discount rate sensitivity
        # table = discount[1]
        # if [ col  for col, dt in table.dtypes.items() if dt == object] != ['']:
        #     table[''] = ["Low discount rate ({:.0f}%)".format(x) for x in table['']]
        # total = total.append(table)

        # # Create table of increased capital costs sensitivity
        # table = cap[0]
        # if [ col  for col, dt in table.dtypes.items() if dt == object] != ['']:
        #     table[''] = ["Increased capital costs by {:.0f}%".format((x-1)*100) for x in table['']]
        # total = total.append(table)

        # # Create table of decreased capital costs sensitivity
        # table = cap[1]
        # if [ col  for col, dt in table.dtypes.items() if dt == object] != ['']:
        #     table[''] = ["Decreased capital costs by {:.0f}%".format(abs((x-1)*100)) for x in table['']]
        # total = total.append(table)

        # # Create table of increased operating costs sensitivity
        # table = op[0]
        # if [ col  for col, dt in table.dtypes.items() if dt == object] != ['']:
        #     table[''] = ["Increased maintenence costs by {:.0f}%".format((x-1)*100) for x in table['']]
        # total = total.append(table)

        # # Create table of decreased opersating costs sensitivity
        # table = op[1]
        # if [ col  for col, dt in table.dtypes.items() if dt == object] != ['']:
        #     table[''] = ["Decreased maintenance costs by {:.0f}%".format(abs((x-1)*100)) for x in table['']]
        # total = total.append(table)

        # # Create table of increased demand sensitivity
        # table = demand[0]
        # if [ col  for col, dt in table.dtypes.items() if dt == object] != ['']:
        #     table[''] = ["Increased demand {:.0f}%".format((x-1)*100) for x in table['']]
        # total = total.append(table)          
        
        # # Create table of decreased demand sensitivity
        # table = demand[1]
        # if [ col  for col, dt in table.dtypes.items() if dt == object] != ['']:
        #     table[''] = ["Decreased demand by {:.0f}%".format(abs((x-1)*100)) for x in table['']]
        # total = total.append(table)

        # # Create table of increased trip distance sensitivity
        # table = trip[0]
        # if [ col  for col, dt in table.dtypes.items() if dt == object] != ['']:
        #     table[''] = ["Increased trip distance by {:.0f}%".format((x-1)*100) for x in table['']]
        # total = total.append(table)

        # # Create table of decreased trip distance sensitivity
        # table = trip[1]
        # if [ col  for col, dt in table.dtypes.items() if dt == object] != ['']:
        #     table[''] = ["Decreased trip distance by {:.0f}%".format(abs((x-1)*100)) for x in table['']]
        # total = total.append(table)

        # # Create table of higher proportion of new trips sensitivity
        # table = new[0]
        # if [ col  for col, dt in table.dtypes.items() if dt == object] != ['']:
        #     table[''] = ["Increased newly degenerated by {:.0f}%".format((x-1)*100) for x in table['']]
        # total = total.append(table)
            
        # # Create table of lower proportion of new trips sensitivity
        # table = new[1]
        # if [ col  for col, dt in table.dtypes.items() if dt == object] != ['']:
        #     table[''] = ["Decreased newly degenerated by {:.0f}%".format(abs((x-1)*100)) for x in table['']]
        # total = total.append(table)

        # # Create table of higher transport trip purpose
        # table = transport[0]
        # if [ col  for col, dt in table.dtypes.items() if dt == object] != ['']:
        #     table[''] = ["Increased work (employer's business trips) {:.0f}%".format(x) for x in table['']]
        # total = total.append(table)

        # # Create table for lower transport trip purpose
        # table = transport[1]
        # if [ col  for col, dt in table.dtypes.items() if dt == object] != ['']:
        #     table[''] = ["Decreased work (employer's business trips) by {:.0f}%".format(abs(x)) for x in table['']]
        # total = total.append(table)

        # # Create basic table style
        # table_style = reportlab.platypus.TableStyle([('FONT', (0,0), (-1,0), 'Helvetica'),
        #          ('LINEABOVE',(0,0),(3,0),1,reportlab.lib.colors.black),
        #          ('LINEBELOW',(0,0),(3,0),0.5,reportlab.lib.colors.black),
        #          ('FONTSIZE', (1,0), (1, 5), 10),
        #          ('ALIGN', (1, 1), (3, 15), "RIGHT"),
        #          ('LINEBELOW',(0,-1),(3,-1),1,reportlab.lib.colors.black)])

        # # Change result values to float and list format
        # total["NPV $000"] = pd.to_numeric(total["NPV $000"], downcast="float")
        # total["BCR1"] = pd.to_numeric(total["BCR1"], downcast="float")
        # total["BCR2"] = pd.to_numeric(total["BCR2"], downcast="float")
        # total =  total.values.tolist()

        # # Set row value to 1
        # row = 1

        # # For each row of the table:
        # for list in total:
                
        #     # Start at column 0
        #     column = 0
                
        #     # For each value in the row:
        #     for number in list:
                    
        #         # If the value is a string:
        #         if isinstance(number, str):

        #             # Skip
        #             pass

        #         # If the value is NPV:
        #         elif column == 1:

        #             # If the NPV is below 0:
        #             if number < 0:

        #                 # Colour red
        #                 table_style.add('TEXTCOLOR', (column, row), (column, row), reportlab.lib.colors.red)
                        
        #             # If the NPV is above 0:
        #             elif number > 0:

        #                 # Colour green
        #                 table_style.add('TEXTCOLOR', (column, row), (column, row), reportlab.lib.colors.green)
                    
        #         # Otherwise
        #         else: 

        #             # If the BCR value is below 1:
        #             if number < 1:

        #                 # Colour red
        #                 table_style.add('TEXTCOLOR', (column, row), (column, row), reportlab.lib.colors.red)
                        
        #             # If the BCR value is above 1:
        #             elif number > 1:

        #                 # Colour green
        #                 table_style.add('TEXTCOLOR', (column, row), (column, row), reportlab.lib.colors.green)
        #         column += 1
        #     row += 1

        # # Convert to dataframe, add column nams and convert to output formats
        # total = pd.DataFrame(total)
        # total.columns = ['', 'NPV $000', 'BCR1', 'BCR2']
        # total['NPV $000'] = ['{:,.0f}'.format(x) for x in (total['NPV $000']/1000)]
        # total['BCR1'] = ['{:.2f}'.format(x) for x in total['BCR1']]
        # total['BCR2'] = ['{:.2f}'.format(x) for x in total['BCR2']]

        # # Create table in reportlab
        # lista = [total.columns[:,].values.astype(str).tolist()] + total.values.tolist()
        # table = reportlab.platypus.Table(lista)
        # table.setStyle(table_style)
        # self.story.append(table)
        # self.story.append(reportlab.platypus.Spacer(1,0.2*reportlab.lib.units.inch))
            
        # Build document
        doc.build(self.story, onFirstPage=myFirstPage, onLaterPages=myLaterPages)
        # return total
    
def write_pdf():
    # Create PDF Document and add results
    result = MyPdfDocument(inputs.facility_name+' Cost-Benefit-Analysis.pdf')
    result.generate_style()
    result.result()
    # total = result.result(discounted_costs, discounted_benefits, input, basic_demand, discount_rate, capex_sensitivity, opex_sensitivity, demand_sensitivity, trip_distance_sensitivity, new_trips_sensitivity, transport_share_sensitivity, divert)
    file = open(inputs.facility_name+' Cost-Benefit-Analysis.pdf', 'rb+')
    reader = PyPDF2.PdfFileReader(file)
    writer = PyPDF2.PdfFileWriter()
    writer.appendPagesFromReader(reader)
    metadata = reader.getDocumentInfo()
    writer.addMetadata(metadata)
    writer.addMetadata({
        '/Author': 'Curtis Syrett',
        '/Title': inputs.facility_name+' Cost-Benefit-Analysis.pdf',
        '/Subject' : 'Economic Research and Analysis',
        '/Producer' : "CropPDF",
        '/Creator' : "CropPDF",
    })
    writer.write(file)
    file.close()
    # return total