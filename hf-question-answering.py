from transformers import pipeline


def main():
    context = """
        The solar system is our cosmic neighborhood, a vast collection of eight planets, dwarf planets, moons, asteroids, comets, and dust. It all revolves around a single star, our sun, which holds everything together with its immense gravity.
        The solar system is estimated to be 4.6 billion years old. It formed from a giant, spinning cloud of gas and dust called a solar nebula. As the nebula collapsed, most of the material coalesced in the center, forming the sun. The leftover material flattened into a disk around the sun, and over time, clumps within this disk clumped together to form the planets and other objects in our solar system.
        The solar system can be broadly divided into two regions: the inner solar system and the outer solar system.
        The inner solar system is home to the four rocky planets: Mercury, Venus, Earth, and Mars. These planets are relatively small and dense, and they all have a solid surface made up of rock and metal.
        The outer solar system is home to the four gas giants: Jupiter, Saturn, Uranus, and Neptune. These planets are much larger and less dense than the inner planets, and they are mostly made up of gas and ice.
        The outer solar system also includes the Kuiper Belt, a vast reservoir of icy objects beyond Neptune's orbit. The Kuiper Belt is home to Pluto, the most famous dwarf planet in our solar system. Other dwarf planets in our solar system include Eris, Haumea, Makemake, and Ceres.
        The solar system is an amazing place, and we are still learning new things about it all the time. With continued exploration, we may one day find evidence of life beyond Earth.    
    """

    quan_pipeline = pipeline("question-answering", model="deepset/minilm-uncased-squad2")
    question = "What are the two regions in the solar system?"
    answer = quan_pipeline(question=question, context=context)
    print("Question: ", question)
    print("Ans: ", answer['answer'])


if __name__ == "__main__":
    main()
