import pandas as pd
import numpy as np

df = pd.DataFrame({'a': [1,2,3], 'b': [2,4,6]})

def tex_begin(file):
    tex = open(file, "w+")
    header = '\n'.join(['\\documentclass{article}','\\usepackage{float}',
    '\\begin{document}',''])
    tex.write(header)

    tex.close()


def tex_table(
    df, file, addnotes=[], mgroups={}, title='', label='',
    options=pd.DataFrame(),footnotesize='footnotesize'):
    '''
This function writes a table to file. The variable name column will just be the index of the dataframe provided. The column headers will be the column headers. The contents will be the contents of the table.

Inputs:
    df: This is the table you want to print

    name: This is the file name (including path) to write to

Options:

    addnotes: Add notes to the bottom of the table (input is a list; each new element is a new line of comment)

    mgroups: a dictionary that defines both the groups and what is in it. For example, mgroups={'Group 1':1,'Group 2':[2,5],'':[6,8]}. The keys are the group header (must be strings), and values are a list (corresponding to the min and max) or integer that defines the regression columns of group. You must specify a complete set of groups though you can define one as blank (as shown) this will cause that section to not have a header or a line underneath it.

    title: A Latex table caption that will be shown at the top of the table.

    label: A label to be used for refering to table in Latex, e.g. use \\ref{label} to refer to the table

    '''
    ###########################################################################
    ### Basic Setup
    ###########################################################################
    spacedict = {'footnotesize':25,'scriptsize':35,'tiny':45}
    if footnotesize not in spacedict.keys():
        print('This footnote size not recognized, defaulting to footnotesize')
        footnotesize = 'footnotesize'


    # If there are groupings, then we iterate through them
    groupedcols = ''
    groupedlines = ''
    if mgroups:
        for key,value in mgroups.items():
            if key:
                if type(value)==int:
                    groupedcols += '& \multicolumn{1}{c}{%s} ' %key
                    groupedlines += '\cline{%s-%s}\n' %(str(value+1),str(value+1))
                elif type(value)==list:
                    groupedcols += '& \multicolumn{%i}{c}{%s} ' %(len(range(value[0],value[1]+1)),key)
                    groupedlines += '\cline{%s}\n' %('-'.join([str(i+1) for i in value]))
            else:
                if type(value)==int:
                    groupedcols += '& '
                elif type(value)==list:
                    groupedcols += '& '*len(range(value[0],value[1]+1))
        groupedcols += '\\\\'
        groupedlines = groupedlines[:-1]


    header = '\n'.join(['\\begin{table}[H]',
                        f'\caption{{{title}}}',
                        f'\label{{{label}}}',
                        '{',
                        '\def\sym#1{\ifmmode^{#1}\else\(^{#1}\)\\fi}',
                        '\\begin{tabular}{@{\extracolsep{2pt}}l*{%i}{c}@{}}' %df.shape[1],
                        '\hline\hline',
                        groupedcols,
                        groupedlines,
                        ''])


    # Basic header, with symbolic command stolen from estout (of stata)
    # We do extra column spacing just in case there is grouping of variables (adds a space)

    # Add footnotes (if any)
    footnotes = ['\multicolumn{%i}{l}{\\%s %s}' %(df.shape[1]+1,footnotesize,ii) for ii in addnotes]
    if footnotes:
        footnotes = ('\\vspace{-.%iem} \\\\\n' %spacedict[footnotesize]).join(footnotes)
        footnotes = [footnotes]
    footer = '\n'.join(['\hline\hline']+footnotes+['\end{tabular}','}','\end{table}'])



    ###########################################################################
    ### Open and write it
    ###########################################################################
    tex = open(file, "a")

    tex.write(header)
    tex.write(' & '.join([''] + df.columns.to_list())+
              '\n'.join([' \\\\','\hline','']))

    for ii in range(0,df.shape[0]):
        row = ' '.join([str(df.index[ii]),'& '])
        for jj in range(0,df.shape[1]):
            row += str(df.iloc[ii,jj])
            if jj<df.shape[1]-1:
                row += ' & '
            else:
                row += '\n'.join([' \\\\',''])
        tex.write(row)

    if options.empty==False:
        tex.write('\n\hline\n')
        for ii in range(0,options.shape[0]):
            row = ' '.join([str(options.index[ii]),'& '])
            for jj in range(0,options.shape[1]):
                row += str(options.iloc[ii,jj])
                if jj<options.shape[1]-1:
                    row += ' & '
                else:
                    row += '\n'.join([' \\\\',''])
            tex.write(row)



    tex.write(footer)
    tex.close()

def tex_end(file):
    tex = open(file, "a")
    header = '\n'.join(['','\\end{document}'])
    tex.write(header)

    tex.close()

filepath = os.path.join(start.results_path, 'memo_what_did_you_hear',
'what_did_you_hear.tex')




tex_begin(filepath)
tex_table(df, filepath)
tex_end(filepath)