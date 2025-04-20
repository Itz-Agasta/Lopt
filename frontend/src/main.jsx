import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "./index.css";
import { GlobalContextProvider } from "./hooks/GlobalContext.jsx";

import { RouterProvider } from "react-router";

import router from "./components/routes.jsx";
import CustomCursor from "./components/utility/CustomCursor.jsx";

createRoot(document.getElementById("root")).render(
  <StrictMode>
    <GlobalContextProvider>
      <RouterProvider router={router}></RouterProvider>
    </GlobalContextProvider>
  </StrictMode>
);
