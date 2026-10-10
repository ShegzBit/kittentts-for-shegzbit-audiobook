import soundfile as sf
from kittentts import KittenTTS

print("Loading KittenTTS 80M model...")
model = KittenTTS("KittenML/kitten-tts-mini-0.8")

print("Available voices:", model.available_voices)

text = (
   """19:54, 15 July, 20XY.

Meerut, Uttar Pradesh, India.

[A Certain God is cheering for you.]

[The Eternal Devotee of Rama is saying that he believes that you would do good.]

[The Remover of Obstacles is showering you with fortune.]

[The Divine Healer of Mankind is nodding at you.]

[The Divine Arbiter of Justice and Order is wishing you luck.]

[The Enchantress of Love is smiling at you.]

[The Metal Detector is placing his bet on your flawless victory.]

[The Dragon Warrior is eager to see another good battle.]

Alex, who could see the new flashes of notification, didn't focus on them as his eyes remained fixed on the sight in front of him.

The place was filled with broken-down buildings and debris everywhere, but there was one strange thing about it.

After every little distance, there was a bright purple-colored fire burning fiercely.

Alex had tried to look around, and he found out that this fire was a type of marker placed there by someone or something.

It was all around this zone—this level 20 zone that he was about to enter.

From the gods' messages, Alex could tell that this fight was not going to be an easy one, but he had already expected this; he was trying to clear a level 20 zone with level 18 stats after all.

He had already hit the level 18 mark that he wanted before he was supposed to try out a level 20 zone, and this was the one closest to him the moment he upgraded himself to level 18, so he decided to raid this zone.

Unlike the previous zones, this one felt more eerie, almost making him turn around and leave, but the fact that the highest stat one could have in this zone was 20 brought him some relief.

Level 20 was not something you would find a lot in a level 20 zone.

No matter which zone we talk about, be it level 5, level 10, or level 15, each of these zones has at least one and at most 2 beasts that have one or two of their stats at maximum, not once had he seen a zone filled with beasts with their levels capped to the highest.

Even if there were many beasts with their levels capped at 20, then he would just come out of the zone unnoticed.."""
)

print("Generating speech...")
audio = model.generate(text, voice="Bella", speed=1.0)

sf.write("output.wav", audio, 24000)
print("Success! Audio saved to output.wav")
