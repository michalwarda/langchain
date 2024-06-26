defmodule LangChain.ChatModels.ChatAnthropicTest do
  use LangChain.BaseCase

  doctest LangChain.ChatModels.ChatAnthropic
  alias LangChain.ChatModels.ChatAnthropic
  alias LangChain.Chains.LLMChain
  alias LangChain.Message

  describe "new/1" do
    test "works with minimal attr" do
      assert {:ok, %ChatAnthropic{} = anthropic} =
               ChatAnthropic.new(%{"model" => "claude-3-opus-20240229"})

      assert anthropic.model == "claude-3-opus-20240229"
    end

    test "returns error when invalid" do
      assert {:error, changeset} = ChatAnthropic.new(%{"model" => nil})
      refute changeset.valid?
      assert {"can't be blank", _} = changeset.errors[:model]
    end

    test "supports overriding the API endpoint" do
      override_url = "http://localhost:1234/v1/messages"

      model =
        ChatAnthropic.new!(%{
          endpoint: override_url
        })

      assert model.endpoint == override_url
    end
  end

  describe "for_api/3" do
    test "generates a map for an API call" do
      {:ok, anthropic} =
        ChatAnthropic.new(%{
          "model" => "claude-3-opus-20240229",
          "temperature" => 1,
          "top_p" => 0.5,
          "api_key" => "api_key"
        })

      data = ChatAnthropic.for_api(anthropic, [])
      assert data.model == "claude-3-opus-20240229"
      assert data.temperature == 1
      assert data.top_p == 0.5
    end

    test "generates a map for an API call with max_tokens set" do
      {:ok, anthropic} =
        ChatAnthropic.new(%{
          "model" => "claude-3-opus-20240229",
          "temperature" => 1,
          "top_p" => 0.5,
          "max_tokens" => 1234
        })

      data = ChatAnthropic.for_api(anthropic, [])
      assert data.model == "claude-3-opus-20240229"
      assert data.temperature == 1
      assert data.top_p == 0.5
      assert data.max_tokens == 1234
    end
  end

  describe "do_process_response/1" do
    test "handles receiving a message" do
      response = %{
        "id" => "id-123",
        "type" => "message",
        "role" => "assistant",
        "content" => [%{"type" => "text", "text" => "Greetings!"}],
        "model" => "claude-3-haiku-20240307",
        "stop_reason" => "end_turn"
      }

      assert %Message{} = struct = ChatAnthropic.do_process_response(response)
      assert struct.role == :assistant
      assert struct.content == "Greetings!"
      assert is_nil(struct.index)
    end

    test "handles receiving a content_block_start event" do
      response = %{
        "type" => "content_block_start",
        "index" => 0,
        "content_block" => %{"type" => "text", "text" => ""}
      }

      assert %MessageDelta{} = struct = ChatAnthropic.do_process_response(response)
      assert struct.role == :assistant
      assert struct.content == ""
      assert is_nil(struct.index)
    end

    test "handles receiving a content_block_delta event" do
      response = %{
        "type" => "content_block_delta",
        "index" => 0,
        "delta" => %{"type" => "text_delta", "text" => "Hello"}
      }

      assert %MessageDelta{} = struct = ChatAnthropic.do_process_response(response)
      assert struct.role == :assistant
      assert struct.content == "Hello"
      assert is_nil(struct.index)
    end

    test "handles receiving a message_delta event" do
      response = %{
        "type" => "message_delta",
        "delta" => %{"stop_reason" => "end_turn", "stop_sequence" => nil},
        "usage" => %{"output_tokens" => 47}
      }

      assert %MessageDelta{} = struct = ChatAnthropic.do_process_response(response)
      assert struct.role == :assistant
      assert struct.content == ""
      assert struct.status == :complete
      assert is_nil(struct.index)
    end
  end

  describe "call/2" do
    @tag live_call: true, live_anthropic: true
    test "handles when invalid API key given" do
      {:ok, chat} = ChatAnthropic.new(%{stream: true, api_key: "invalid"})

      {:error, reason} =
        ChatAnthropic.call(chat, [
          Message.new_user!("Return the response 'Colorful Threads'.")
        ])

      assert reason == "Authentication failure with request"
    end

    @tag live_call: true, live_anthropic: true
    test "basic streamed content example" do
      {:ok, chat} = ChatAnthropic.new(%{stream: true})

      {:ok, result} =
        ChatAnthropic.call(chat, [
          Message.new_user!("Return the response 'Colorful Threads'.")
        ])

      # returns a list of MessageDeltas.
      assert result == [
               [
                 %LangChain.MessageDelta{
                   content: "",
                   status: :incomplete,
                   index: nil,
                   function_name: nil,
                   role: :assistant,
                   arguments: nil
                 }
               ],
               [
                 %LangChain.MessageDelta{
                   content: "Color",
                   status: :incomplete,
                   index: nil,
                   function_name: nil,
                   role: :assistant,
                   arguments: nil
                 }
               ],
               [
                 %LangChain.MessageDelta{
                   content: "ful",
                   status: :incomplete,
                   index: nil,
                   function_name: nil,
                   role: :assistant,
                   arguments: nil
                 }
               ],
               [
                 %LangChain.MessageDelta{
                   content: " Threads",
                   status: :incomplete,
                   index: nil,
                   function_name: nil,
                   role: :assistant,
                   arguments: nil
                 }
               ],
               [
                 %LangChain.MessageDelta{
                   content: "",
                   status: :complete,
                   index: nil,
                   function_name: nil,
                   role: :assistant,
                   arguments: nil
                 }
               ]
             ]
    end
  end

  describe "decode_stream/1" do
    test "when data is broken" do
      data1 = ~s|event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hr"}       }\n\n
event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"asing"}      }\n\n
event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" back"}           }\n\n
event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" what"}             }\n\nevent: content_block_delta\ndata: {"type":"content_block_delta","index":0|

      {processed1, incomplete} = ChatAnthropic.decode_stream({data1, ""})

      assert incomplete ==
               ~s|event: content_block_delta\ndata: {"type":"content_block_delta","index":0|

      assert processed1 == [
               %{
                 "delta" => %{"text" => "hr", "type" => "text_delta"},
                 "index" => 0,
                 "type" => "content_block_delta"
               },
               %{
                 "delta" => %{"text" => "asing", "type" => "text_delta"},
                 "index" => 0,
                 "type" => "content_block_delta"
               },
               %{
                 "delta" => %{"text" => " back", "type" => "text_delta"},
                 "index" => 0,
                 "type" => "content_block_delta"
               },
               %{
                 "delta" => %{"text" => " what", "type" => "text_delta"},
                 "index" => 0,
                 "type" => "content_block_delta"
               }
             ]

      data2 = ~s|,"delta":{"type":"text_delta","text":" your"}    }\n\n
event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" friend"}               }\n\n
event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" said"}   }\n\n|

      {processed2, incomplete} = ChatAnthropic.decode_stream({data2, incomplete})

      assert incomplete == ""

      assert processed2 == [
               %{
                 "delta" => %{"text" => " your", "type" => "text_delta"},
                 "index" => 0,
                 "type" => "content_block_delta"
               },
               %{
                 "delta" => %{"text" => " friend", "type" => "text_delta"},
                 "index" => 0,
                 "type" => "content_block_delta"
               },
               %{
                 "delta" => %{"text" => " said", "type" => "text_delta"},
                 "index" => 0,
                 "type" => "content_block_delta"
               }
             ]
    end

    test "can parse streaming events" do
      chunk = """
      event: message_start
      data: {"type":"message_start","message":{"id":"msg_01CsrHBjq3eHRQjYG5ayuo5o","type":"message","role":"assistant","content":[],"model":"claude-3-sonnet-20240229","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":14,"output_tokens":1}}}

      event: content_block_start
      data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

      event: ping
      data: {"type": "ping"}

      event: content_block_delta
      data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}

      """

      {parsed, buffer} = ChatAnthropic.decode_stream({chunk, ""})

      assert [
               %{
                 "type" => "content_block_start",
                 "content_block" => %{"text" => "", "type" => "text"},
                 "index" => 0
               },
               %{
                 "type" => "content_block_delta",
                 "delta" => %{"text" => "Hello", "type" => "text_delta"},
                 "index" => 0
               }
             ] = parsed

      assert buffer == ""

      chunk = """
      event: content_block_delta
      data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"!"}}

      """

      {parsed, buffer} = ChatAnthropic.decode_stream({chunk, ""})

      assert [
               %{
                 "type" => "content_block_delta",
                 "delta" => %{"text" => "!", "type" => "text_delta"},
                 "index" => 0
               }
             ] = parsed

      assert buffer == ""

      chunk = """
      event: content_block_stop
      data: {"type":"content_block_stop","index":0}

      event: message_delta
      data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens": 3}}

      event: message_stop
      data: {"type":"message_stop"}

      """

      {parsed, buffer} = ChatAnthropic.decode_stream({chunk, ""})

      assert [
               %{
                 "type" => "message_delta",
                 "delta" => %{"stop_reason" => "end_turn", "stop_sequence" => nil},
                 "usage" => %{"output_tokens" => 3}
               }
             ] = parsed

      assert buffer == ""
    end

    test "non-ascii unicode character (en dash U+2013)" do
      chunk = """
      event: content_block_delta
      data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" –"}}

      event: content_block_delta
      data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" Anthrop"}}

      """

      {parsed, buffer} = ChatAnthropic.decode_stream({chunk, ""})

      assert [
               %{
                 "type" => "content_block_delta",
                 "index" => 0,
                 "delta" => %{"type" => "text_delta", "text" => " –"}
               },
               %{
                 "type" => "content_block_delta",
                 "index" => 0,
                 "delta" => %{"type" => "text_delta", "text" => " Anthrop"}
               }
             ] = parsed

      assert buffer == ""
    end

    test "handles incomplete chunks" do
      chunk_1 =
        "event: content_blo"

      {parsed, buffer} = ChatAnthropic.decode_stream({chunk_1, ""})

      assert [] = parsed
      assert buffer == chunk_1

      chunk_2 =
        "ck_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"de"

      {parsed, buffer} = ChatAnthropic.decode_stream({chunk_2, buffer})

      assert [] = parsed
      assert buffer == chunk_1 <> chunk_2

      chunk_3 = ~s|lta":{"type":"text_delta","text":"!"}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" Anthrop"}}

|

      {parsed, buffer} = ChatAnthropic.decode_stream({chunk_3, buffer})

      assert [
               %{
                 "type" => "content_block_delta",
                 "delta" => %{"text" => "!", "type" => "text_delta"},
                 "index" => 0
               },
               %{
                 "type" => "content_block_delta",
                 "delta" => %{"text" => " Anthrop", "type" => "text_delta"},
                 "index" => 0
               }
             ] = parsed

      assert buffer == ""
    end
  end

  describe "works within a chain" do
    @tag live_call: true, live_anthropic: true
    test "works with a streaming response" do
      {:ok, _result_chain, last_message} =
        LLMChain.new!(%{llm: %ChatAnthropic{stream: true}})
        |> LLMChain.add_message(Message.new_user!("Say, 'Hi!'!"))
        |> LLMChain.run()

      assert last_message.content == "Hi!"
      assert last_message.status == :complete
      assert last_message.role == :assistant
    end

    @tag live_call: true, live_anthropic: true
    test "works with NON streaming response" do
      {:ok, _result_chain, last_message} =
        LLMChain.new!(%{llm: %ChatAnthropic{stream: false}})
        |> LLMChain.add_message(Message.new_user!("Say, 'Hi!'!"))
        |> LLMChain.run()

      assert last_message.content == "Hi!"
      assert last_message.status == :complete
      assert last_message.role == :assistant
    end
  end
end
