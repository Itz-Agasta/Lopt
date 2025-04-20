import fake_image_1 from "../../assets/df/fake_image_1.jpg";
import fake_image_2 from "../../assets/df/fake_image_2.jpg";
import fake_video_1 from "../../assets/df/fake_video_1.mp4";
import real_image_1 from "../../assets/df/real_image_1.jpg";
import real_image_2 from "../../assets/df/real_image_2.jpg";
import real_video_1 from "../../assets/df/real_video_1.mp4";

const samples = [
    {   source: fake_image_1,
        name: "fake_image_1"
    },
    {   source: fake_image_2,
        name: "fake_image_2"
    },
    {   source: fake_video_1,
        name: "fake_video_1"
    },
    {   source: real_image_1,
        name: "real_image_1"
    },
    {   source: real_image_2,
        name: "real_image_2"
    },
    {   source: real_video_1,
        name: "real_video_1"
    },
];

export default samples;