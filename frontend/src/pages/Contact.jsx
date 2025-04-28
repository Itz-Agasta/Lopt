import NavBar from "../components/utility/NavBar";
import Footer from "../components/utility/Footer";
import background from "../assets/bg.png";
import github from "../assets/github.svg";
import { motion } from "framer-motion";
import { useGlobalContext } from "../hooks/GlobalContext";

function Contact() {
  const { isMenuOpen } = useGlobalContext();

  return (
    <>
      <div className="h-screen w-screen">
        <NavBar />
        <img className="absolute h-screen w-screen z-0" src={background}></img>

        <div className="relative z-10 w-full h-full flex justify-center items-center px-6">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="bg-[#1e1e1e]/80 backdrop-blur-md p-8 rounded-2xl w-full max-w-3xl border border-[#2a2a2a]"
          >
            <h2 className="text-white text-2xl md:text-4xl font-[muBold] mb-6">
              Contact Us
            </h2>
            <p className="text-[#c2c2c2] mb-6 text-sm md:text-md">
              Got questions or suggestions? Reach out and we'll get back to you
              as soon as possible.
            </p>
            <form className="flex flex-col space-y-4">
              <input
                type="text"
                placeholder="Your Name"
                className="bg-[#2b2b2b] text-white p-3 rounded-lg outline-none focus:ring-2 ring-[#f03b05]"
              />
              <input
                type="email"
                placeholder="Your Email"
                className="bg-[#2b2b2b] text-white p-3 rounded-lg outline-none focus:ring-2 ring-[#f03b05]"
              />
              <textarea
                placeholder="Your Message"
                className="bg-[#2b2b2b] text-white p-3 rounded-lg h-32 resize-none outline-none focus:ring-2 ring-[#f03b05]"
              ></textarea>
              <button
                type="submit"
                className="bg-[#f03b05] hover:bg-[#cb3105] transition-colors duration-300 text-white p-3 rounded-lg font-semibold"
              >
                Send Message
              </button>
            </form>
          </motion.div>
        </div>
        <div className="fixed z-20 md:left-5 md:top-[70%] right-5 bottom-[15%] flex flex-col md:w-[2vw]">
          <button className="bg-[#1E1E1E] p-1">
            <a href="https://github.com/Itz-Agasta/Lopt.git">
              <img src={github} height={10} width={20} alt="GitHub" />
            </a>
          </button>
        </div>
        <Footer />
      </div>
    </>
  );
}

export default Contact;
