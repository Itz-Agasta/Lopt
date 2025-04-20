import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "./index.css";
import { GlobalContextProvider } from "./hooks/GlobalContext.jsx";

import { RouterProvider } from "react-router";

import router from "./components/routes.jsx";
import { ReactLenis } from "./components/utility/lenis.jsx";

createRoot(document.getElementById("root")).render(
  <StrictMode>
    <GlobalContextProvider>
      <RouterProvider router={router}></RouterProvider>
    </GlobalContextProvider>
  </StrictMode>
);
