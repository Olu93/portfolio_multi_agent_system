import Particles, { initParticlesEngine } from "@tsparticles/react";
import { loadSlim } from "@tsparticles/slim"; // Updated to loadSlim for smaller bundle size
import { useEffect, useMemo, useState } from "react";

export default function ParticlesBackground() {
    const [init, setInit] = useState(false);

    // Particles initialization
    useEffect(() => {
      initParticlesEngine(async (engine) => {
        await loadSlim(engine);
      }).then(() => {
        setInit(true);
      });
    }, []);
  
    const particlesOptions = useMemo(
      () => ({
        fullScreen: {
          enable: true,
          zIndex: -1,
        },
        background: {
          color: {
            value: "#000000",
          },
        },
  
        particles: {
          number: {
            value: 50,
          },
          color: {
            value: "#ffffff",
          },
          shape: {
            type: "circle",
          },
          opacity: {
            value: 0.5,
          },
          size: {
            value: { min: 1, max: 5 },
          },
          move: {
            enable: true,
            speed: 2,
          },
        },
        interactivity: {
          events: {
            onHover: {
              enable: true,
              mode: "repulse",
            },
          },
        },
        detectRetina: true,
      }),
      []
    );
    return (<span>{init && <Particles id="tsparticles" options={particlesOptions} />}</span>)

}