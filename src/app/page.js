import Navbar from '../components/Navbar'

export default function Home() {
  return (
    <>
      <Navbar />

      {/* Main Landing Section */}
      <div className="flex flex-col justify-center items-center min-h-screen bg-gradient-to-r from-indigo-500 to-purple-600">
        <h1 className="text-5xl font-bold text-white mb-4">Hi, I&apos;m [Your Name]</h1>
        <p className="text-xl text-white">A Passionate Developer & Analyst 🚀</p>
      </div>

      {/* About Section */}
      <section id="about" className="p-10 bg-white text-black text-center">
        <h2 className="text-3xl font-bold mb-2">About Me</h2>
        <p>I am a software developer skilled in Python, AWS, Power BI, SQL, DevOps, and AI/ML.</p>
      </section>

      {/* Projects Section */}
      <section id="projects" className="p-10 bg-gray-100 text-black text-center">
        <h2 className="text-3xl font-bold mb-4">Projects</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="p-4 border rounded shadow bg-white">
            <h3 className="text-xl font-semibold mb-2">Real-Time Emotion Recognition</h3>
            <p>AI system detecting emotions using facial expressions & speech in real-time.</p>
          </div>
          <div className="p-4 border rounded shadow bg-white">
            <h3 className="text-xl font-semibold mb-2">AWS DevOps Automation Pipeline</h3>
            <p>Automated CI/CD pipeline using AWS, Terraform, Kubernetes, Jenkins.</p>
          </div>
          <div className="p-4 border rounded shadow bg-white">
            <h3 className="text-xl font-semibold mb-2">AI-Driven Social Media Analysis</h3>
            <p>Analyzing public sentiment using NLP models and visualizing trends.</p>
          </div>
        </div>
      </section>

      {/* Contact Section */}
      <section id="contact" className="p-10 bg-white text-black text-center">
        <h2 className="text-3xl font-bold mb-2">Contact Me</h2>
        <p>Email: yourname@example.com</p>
        <p>LinkedIn: <a href="https://linkedin.com/in/yourname" target="_blank" className="text-blue-500">Visit Profile</a></p>
      </section>
    </>
  )
}
