import obama from "../../assets/images/obama.jpg";
import model1 from "../../assets/virtus.png";
import model2 from "../../assets/scarlett.png";
import { NavLink } from "react-router";
import arrow from "../../assets/arrow2.svg";
import arrow2 from "../../assets/arrow3.svg";
import Footer from "./Footer";
import Reveal1 from "./Reveal1";
import Reveal2 from "./Reveal2";

const Details = () => {
  return (
    <>
      <div className="absolute top-[100%] w-[100vw] h-[200vh] z-100 bg-black brightness-80">
        <div className="relative min-h-[30vh] w-screen flex flex-col bg-black">
          <Reveal2>
            <div className="font-[mubold] text-5xl text-white w-[60vw] ml-10 mt-5">
              APPROXIMATELY 500,000 DEEPFAKES
              <br />
              SHARED ON SOCIAL-MEDIA IN 2024
              <span className="text-sm">
                <a href="https://artsmart.ai/blog/ai-generated-deepfakes-statistics/">
                  {"  "}[SOURCE]
                </a>
              </span>
            </div>
          </Reveal2>
          <div className="font-[mubold] text-7xl text-[#212020] text-end w-[50vw] ml-[48vw] mt-5 mr-5">
            / BUILT BY
            <br />
            TEAM VYSE
          </div>
        </div>
        <Reveal1>
          <div className="relative w-screen min-h-[50vh] flex justify-center">
            <div className="w-[90%] min-h-[100%] rounded-2xl border-[0.08rem] border-[#232322] flex flex-row space-x-5">
              <div className="min-h-[100%] text-start text-white text-5xl mt-10 ml-8">
                (01)
              </div>
              <div className="flex justify-center items-center">
                <div className="w-[350px] h-[350px] flex justify-center items-center">
                  <img src={obama} height={300} width={300}></img>
                </div>
              </div>
              <div className="flex flex-col justify-center items-center p-5">
                <p className="text-2xl text-white">
                  <span className="text-sm p-5">(About)</span>The deepfake video
                  of former President Barack Obama was produced and released in
                  April 2018. It was created by actor and director Jordan Peele
                  in collaboration with BuzzFeed and Monkeypaw Productions.
                </p>
                <div className="w-[70%] mt-5">
                  <div className="border-t-[0.08rem] border-b-[0.08rem] border-[#232322] text-white text-xl flex flex-row justify-between p-5">
                    <span className="text-[#f03b05]">01.</span>
                    <p>Virtus</p>
                  </div>
                  <div className="border-t-[0.08rem] border-b-[0.08rem] border-[#232322] text-white text-xl flex flex-row justify-between p-5">
                    <span className="text-[#f03b05]">02.</span>
                    <p>Scarlett</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </Reveal1>
        <div className="relative min-h-[20vh] w-screen flex bg-black mt-10 justify-end">
          <p className="text-5xl text-white font-[mubold] ml-5">
            Where reality meets AI integrity
          </p>
        </div>
        <div className="relative min-h-[50vh] w-screen flex flex-row bg-black mt-10 justify-between">
          <Reveal2>
            <div className="min-h-[10vh] ml-5">
              <p className="text-5xl text-white font-[mubold] ml-5">
                OUR MODELS
              </p>
              <p className="text-xl text-white font-[mubold] ml-5 max-w-[40vw] mt-10">
                Lopt is a multimodal deepfake detection framework designed to
                analyze and classify images and videos as Real or Fake. It
                combines: ViT-based architectures for image classification
                TimeSformer-based models for video understanding Clean and
                reproducible pipelines for training, evaluation, and deployment
                Full-stack implementation for both research and real-world use
              </p>
              <NavLink to="/playground">
                <button className="bg-[#1E1E1E] h-[3.4rem] w-[12rem] flex flex-row justify-between items-center space-x-4 rounded-xs rounded-br-2xl mt-2 px-1 group">
                  <div className="flex flex-row justify-center items-center space-x-2 min-w-full h-[90%] rounded-xs rounded-br-2xl after:transition-all after:duration-400 group-hover:bg-[#f03b05] transition-colors duration-500">
                    <span
                      href="/playground"
                      className="relative inline-block w-[80%]"
                    >
                      <span className="after:content-[''] after:absolute after:left-2 after:bottom-0 after:h-[0.08rem] after:w-0 after:bg-white after:transition-all after:duration-500 group-hover:after:w-[85%] text-sm text-white inter-400">
                        Try Lopt Now!
                      </span>
                    </span>
                    <div className="bg-[#606060] h-[3rem] w-[3rem] flex flex-col justify-center items-center rounded-br-2xl ml-[0.4rem] group-hover:bg-[#cb3105] transition-colors duration-300">
                      <img
                        src={arrow}
                        height={12}
                        width={12}
                        className="absolute group-hover:hidden"
                      ></img>
                      <img
                        src={arrow2}
                        height={12}
                        width={12}
                        className="opacity-0 group-hover:opacity-100"
                      ></img>
                    </div>
                  </div>
                </button>
              </NavLink>
            </div>
          </Reveal2>
          <div className="mr-5 flex flex-row justify-center items-center space-x-10">
            <div className="space-y-3">
              <img
                src={model1}
                height={200}
                width={240}
                className="hover:scale-110 transition-all duration-500"
              ></img>
              <p className="text-xl text-white font-[muBold] text-center mt-5">
                Virtus
              </p>
            </div>
            <div>
              <img
                src={model2}
                height={250}
                width={250}
                className="hover:scale-110 transition-all duration-500"
              ></img>
              <p className="text-xl text-white font-[muBold] text-center mt-5">
                Scarlett
              </p>
            </div>
          </div>
        </div>
        <Footer />
      </div>
    </>
  );
};

export default Details;
