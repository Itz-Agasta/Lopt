"use client";

const LoadingScreen = ({ text = "LOADING LOPT..." }) => {
  return (
    <div className="fixed z-999 inset-0 flex flex-col items-center justify-center bg-black">
      <style jsx>{`
        @keyframes grow {
          0%,
          100% {
            transform: scaleX(0.05);
          }
          50% {
            transform: scaleX(1);
          }
        }
      `}</style>
      <div className="flex flex-col items-center gap-4">
        {/* Horizontal Progress Bar */}
        <div className="w-64 h-1 bg-[#1E1E1E] rounded-full overflow-hidden">
          <div
            className="h-full w-full bg-white rounded-full origin-left animate-[grow_1.5s_ease-in-out_infinite]"
            style={{
              animation: "grow 3.5s ease-in-out infinite",
              transformOrigin: "left",
            }}
          />
        </div>

        {/* Loading text */}
        <p className="text-lg font-medium text-[#f03b05] font-[mubold]">
          {text}
        </p>
      </div>
    </div>
  );
};

export default LoadingScreen;
