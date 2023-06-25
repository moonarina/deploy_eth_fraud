import streamlit as st

def main():
    #---mengatur lebar tampilan---
    st.markdown('<style>body(max-width: 800 px; margin: auto;)</style>', unsafe_allow_html=True)

    #---title---
    st.title('Arina\'s Portfolio 📂')
    st.markdown('---')

    #---import photo---
    #pic = './images/photo_profile.png'
    #st.image(pic, width=150, caption='photo profile', use_column_width=False, output_format='PNG') #menampilkan foto

    st.header('About Me')
    st.write('Arina is a collaborative person, with a dedication, to partnering with coworkers to promote an engaging, empowering work value system. Documented strengths in building and maintaining relationships with a diverse field of decision-makers in dynamic, fast-paced settings who care about data for the customer and organization. An insightful person with experience directing and slashing operations through effective employee motivational strategies and solid policy enforcement. Proficient in best practices, market trends, and regulatory requirements of industry operations. A talented person with an analytical approach to business planning and day-to-day rebuilding.')
    st.write('## Skills:')
    st.markdown('''
    1. Python programming
    2. Data analytics
    3. Machine learning
    4. Deep learning
    5. Artificial intelligence
    ''')

    st.header('📩 Find Me At:')
    st.markdown('''
    - www.linkedin.com/in/moonarina
    - www.github.com/moonarina
    - arina.data.scientist@gmail.com''')

if __name__ == '__main__':
    main()

