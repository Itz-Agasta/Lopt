import obama from "../../assets/images/obama.jpg";

const Details = () => {
  return (
    <div className="absolute top-[100%] w-[100vw] h-[110vh] z-100 bg-black brightness-80">
      <div className="relative min-h-[30vh] w-screen flex flex-col bg-black">
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
        <div className="font-[mubold] text-7xl text-[#212020] text-end w-[50vw] ml-[48vw] mt-5 mr-5">
          / BUILT BY
          <br />
          TEAM VYSE
        </div>
      </div>
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
              <span className="text-sm p-5">(About)</span>The deepfake video of
              former President Barack Obama was produced and released in April
              2018. It was created by actor and director Jordan Peele in
              collaboration with BuzzFeed and Monkeypaw Productions.
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
    </div>
  );
};

export default Details;
