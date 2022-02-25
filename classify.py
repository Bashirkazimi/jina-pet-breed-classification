import streamlit as st
import helper
from jina import Flow


# function to show image and its predicted breed
def show_pet_and_breed(tags, image):
    """
    Shows an image of a pet and prints out its predicted breed and probability using the tags dictionary
    """
    breed = tags['label'] # the predicted breed
    pet_category = 'cat' if breed[0].isupper() else 'dog' # capitalized breed categories are cats, otherwise dogs
    breed = breed.lower()
    breed = ' '.join(breed.split('_')) # multi-word categories are joined using '_', so replace it with space
    article = 'an' if breed[0] in ['a', 'e', 'i', 'o', 'u'] else 'a' # the definite article for category to be printed!
    st.image(image, caption="I am {} percent sure this is {} {} {}".format(round(tags['prob']*100), article, breed , pet_category))


def main():
    # Layout
    st.set_page_config(page_title="Jina Pet Breed Classification")
    st.markdown(
        body=helper.UI.css,
        unsafe_allow_html=True,
    )

    # Sidebar
    st.sidebar.markdown(helper.UI.about_block, unsafe_allow_html=True)

    st.header("Jina Pet Breed Classification")

    upload_cell, preview_cell = st.columns([12, 1])
    query = upload_cell.file_uploader("")
    if query:
        doc, image = helper.convert_file_to_document(query)
        if st.button(label="Classify"):
            if not query:
                st.markdown("Please upload a pet image")
            else:
                # Flow
                flow = Flow().add(uses='jinahub://PetBreedClassifier')
                try:
                    flow.start()
                    flow.post(on='/index', inputs=doc, on_done=lambda resp: show_pet_and_breed(resp.docs[0].tags, image))
                finally:
                    flow.close()


if __name__ == '__main__':
    main()


