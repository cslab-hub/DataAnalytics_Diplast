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
    
    st.markdown(""" 
        ##### This dashboard is developed by Jheronimus Academy of Data Science (JADS) and Osnabrück University (UOS). It is developed for the Di-Plast project. More information on the Di-Plast project can be found [here](https://www.nweurope.eu/projects/project-search/di-plast-digital-circular-economy-for-the-plastics-industry/). For information on this dedicated tool, we advise reading our [wiki](https://di-plast.sis.cs.uos.de/Wiki.jsp?page=Data%20Analytics) page containing all the necessary information for installation and interpretation.


        """)

    st.markdown(
        """

     












        """)



    st.markdown(
        """
        ##### The Data Analytics tool consists of several modules that analyse parts of your dataset. It is of great importance that the data used in this tool is properly validated. For validating the data, we advise you to check our data validation tool that can be accessed [here](https://cslab-hub-data-validation-main-bx6ggw.streamlitapp.com/).
        ##### **👈 Select a tool from the dropdown menu on the left.**""")
    



  

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
    st.markdown(""" Developed in the Di-Plast project in collaboration with:


        """)

    col1, col2  = st.columns(2)

    with col1:
        st.image('images/JADS_logo.png', width=500)


    with col2:
        st.image('images/Universitat_Osnabruck.png', width=400)




    st.markdown("""
        <div>Icons made by <a href="https://www.flaticon.com/authors/roundicons-premium" title="Roundicons Premium">Roundicons Premium</a> from <a href="https://www.flaticon.com/" title="Flaticon">www.flaticon.com</a></div>
    """,  unsafe_allow_html=True)
