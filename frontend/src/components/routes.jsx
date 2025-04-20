import { createBrowserRouter } from "react-router";
import Home from "../pages/Home";
import Playground from "../pages/Playground";

const router = createBrowserRouter([
  {
    path: "/",
    element: <Home />,
  },
  {
    path: "/home",
    element: <Home />,
  },
  {
    path: "/playground",
    element: <Playground />,
  },
]);

export default router;
