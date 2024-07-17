css = """
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #2b313e
}
.chat-message.bot {
    background-color: #475063
}
.chat-message .avatar {
  width: 20%;
}
.chat-message .avatar img {
  max-width: 78px;
  max-height: 78px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #fff;
}
"""

bot_template = """
<div class="chat-message user">
    <div class="avatar">
        <img src="https://i.ibb.co/rfsQsGL/DSC01771.jpg">
    </div>
    <div class="message"><span style="color: black; font-weight: bold; font-size: large;">User:</span> {{MSG}}</div>
</div>
"""


user_template = """
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://i.ibb.co/h14DWM6/251af2b2-8f29-4f96-8502-3662300f09ce.jpg">
    </div>
    <div class="message"><span style="color: black; font-weight: bold; font-size: large;">Assistant:</span> {{MSG}}</div>
</div>

"""
