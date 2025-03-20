export default function Navbar() {
    return (
      <nav className="flex justify-around items-center bg-black text-white p-4">
        <a href="#about">About</a>
        <a href="#projects">Projects</a>
        <a href="#contact">Contact</a>
        <a href="/Lokesh_Resume.pdf" download className="bg-green-500 text-white p-2 rounded">
  Download Resume
</a>

      </nav>
    )
  }
  