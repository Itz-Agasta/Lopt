import NavBar from "../components/utility/NavBar";
import Footer from "../components/utility/Footer";
import background from "../assets/bg.png";
import github from "../assets/github.svg";
import { motion } from "framer-motion";
import { useGlobalContext } from "../hooks/GlobalContext";
import Sticker from "../components/utility/h4b";

function About() {
  const { isMenuOpen } = useGlobalContext();

  return (
    <>
      <div className="h-screen w-screen">
        <NavBar />
        <img className="absolute h-screen w-screen z-0" src={background}></img>
        {isMenuOpen ? (
          <div className="h-screen w-screen z-20 fixed inset-0 backdrop-brightness-75"></div>
        ) : (
          <></>
        )}
        <div className="relative z-10 w-full h-full flex justify-center items-center px-6">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="bg-[#1e1e1e]/80 backdrop-blur-md p-8 rounded-2xl w-full max-w-3xl border border-[#2a2a2a]"
          >
            <h2 className="text-white text-2xl md:text-4xl font-[muBold] mb-6">
              About Us
            </h2>
            <p className="text-[#c2c2c2] mb-4 text-sm md:text-md">
              We are a passionate group of engineers, designers, and researchers
              from VIT focused on building cutting-edge tools that combat
              deepfakes and misinformation.
            </p>
            <p className="text-[#c2c2c2] mb-4 text-sm md:text-md">
              Our product,{" "}
              <span className="text-white font-semibold">LOPT</span>, is a
              state-of-the-art deepfake detection model trained on a robust
              dataset to ensure high accuracy and transparency.
            </p>
            <p className="text-[#c2c2c2] text-sm md:text-md">
              Our mission is to promote media integrity and empower users with
              the tools to verify content authenticity—quickly and reliably.
            </p>
          </motion.div>
        </div>
        <Sticker />
        <div className="fixed z-20 md:left-5 md:top-[70%] right-5 bottom-[15%] flex flex-col md:w-[2vw]">
          <button className="bg-[#1E1E1E] p-1">
            <a href="https://github.com/Itz-Agasta/Lopt.git">
              <img src={github} height={10} width={20} alt="GitHub" />
            </a>
          </button>
        </div>
      </div>
      <Footer />
    </>
  );
}

export default About;
