import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image 

def return_homepage():
    # col1, col2, col3 = st.columns([1,2.5,1])
    # with col2:
        # image = Image.open('images/logo.jpeg')
        # st.image(image, use_column_width=True)

    st.header('Welcome to the Data Analytics Tool')
    st.markdown(
        """
        #### The Data Analytics tool consists of several modules that analyse parts of your dataset. It is of great importance that the data used in this tool is properly validated. For validating the data, we advise you to check our data validation tool that can be accessed [here](https://share.streamlit.io/cslab-hub/data_validation_diplast/main/main.py) 
        #### **👈 Select a tool from the dropdown menu on the left**""")
    

    # st.markdown(
    #     """
    #     ### Usage of the Data Analytics tool:\n
    #     #### As can be seen on the left side of this page, several different applications can be accessed for analysing your data. However, to actually analyse the data, a datafile should be placed within the "data" folder of this package. 



    #     """)

    st.markdown(
        """
     









        """)


    with st.expander("See Disclaimer"):

        st.write("""
                15. Disclaimer of Warranty.
                THERE IS NO WARRANTY FOR THE PROGRAM, TO THE EXTENT PERMITTED BY APPLICABLE LAW. 
                EXCEPT WHEN OTHERWISE STATED IN WRITING THE COPYRIGHT HOLDERS AND/OR OTHER PARTIES PROVIDE THE PROGRAM “AS IS” WITHOUT WARRANTY OF ANY KIND, EITHER EXPRESSED OR IMPLIED, 
                INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. 
                THE ENTIRE RISK AS TO THE QUALITY AND PERFORMANCE OF THE PROGRAM IS WITH YOU. 
                SHOULD THE PROGRAM PROVE DEFECTIVE, YOU ASSUME THE COST OF ALL NECESSARY SERVICING, REPAIR OR CORRECTION.
                16. Limitation of Liability.
                IN NO EVENT UNLESS REQUIRED BY APPLICABLE LAW OR AGREED TO IN WRITING WILL ANY COPYRIGHT HOLDER, OR ANY OTHER PARTY WHO MODIFIES AND/OR CONVEYS THE PROGRAM AS PERMITTED ABOVE, 
                BE LIABLE TO YOU FOR DAMAGES, INCLUDING ANY GENERAL, SPECIAL, 
                INCIDENTAL OR CONSEQUENTIAL DAMAGES ARISING OUT OF THE USE OR INABILITY TO USE THE PROGRAM (INCLUDING BUT NOT LIMITED TO LOSS OF DATA OR DATA BEING RENDERED INACCURATE OR LOSSES SUSTAINED BY YOU OR THIRD PARTIES OR A FAILURE OF THE PROGRAM TO OPERATE WITH ANY OTHER PROGRAMS), 
                EVEN IF SUCH HOLDER OR OTHER PARTY HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.
                """)

    st.markdown("""
        <div>Icons made by <a href="https://www.flaticon.com/authors/roundicons-premium" title="Roundicons Premium">Roundicons Premium</a> from <a href="https://www.flaticon.com/" title="Flaticon">www.flaticon.com</a></div>
    """,  unsafe_allow_html=True)