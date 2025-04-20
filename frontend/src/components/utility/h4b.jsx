import hack4bengal from "../../assets/hack4b.png";
const Sticker = () => {
  return (
    <>
      <div className="fixed z-9999 md:-left-0 -right-0 md:top-[40%] top-[60%] md:h-[10rem] md:w-[3rem] h-[8rem] w-[2rem] bg-white flex flex-col">
        <img src={hack4bengal} className="-rotate-90"></img>
        <p className="[writing-mode:vertical-rl] rotate-180 text-center md:text-[16px] text-xs ml-2 mt-2 md:ml-3 inter-600">
          hack4bengal
        </p>
      </div>
    </>
  );
};

export default Sticker;
